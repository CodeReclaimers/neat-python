"""
Distributed evaluation of genomes.

About compute nodes:
The primary node (=the node which creates and mutates genomes) and the secondary
nodes (=the nodes which evaluate genomes) can execute the same script. The
role of a compute node is determined using the ``mode`` argument of the
DistributedEvaluator. If the mode is MODE_AUTO, the `host_is_local()` function
is used to check if the ``addr`` argument points to the localhost. If it does,
the compute node starts as a primary node, otherwise as a secondary node. If
``mode`` is MODE_PRIMARY, the compute node always starts as a primary node. If
``mode`` is MODE_SECONDARY, the compute node will always start as a secondary node.

There can only be one primary node per NEAT, but any number of secondary nodes.
The primary node will not evaluate any genomes, which means you will always need
at least two compute nodes.

You can run any number of compute nodes on the same physical machine (or VM).
However, if a machine has both a primary node and one or more secondary nodes,
MODE_AUTO cannot be used for those secondary nodes - MODE_SECONDARY will need to be
specified.

NOTE:
    This module is in a **beta** state, and still *unstable* even in single-machine testing. Reliability is likely to vary, including depending on the Python version
    and implementation (e.g., cpython vs pypy) in use and the likelihoods of timeouts (due to machine and/or network slowness). In particular, while the code can try
    to reconnect between between primary and secondary nodes, as noted in the `multiprocessing` documentation this may not work due to data loss/corruption. Note also
    that this module is not responsible for starting the script copies on the different compute nodes, since this is very site/configuration-dependent.


Usage:
1. Import modules and define the evaluation logic (the eval_genome function).
  (After this, check for ``if __name__ == '__main__'``, and put the rest of
  the code inside the body of the statement.)
2. Load config and create a population - here, the variable ``p``.
3. If required, create and add reporters.
4. Create a ``DistributedEvaluator(addr_of_primary_node, b'some_password',
  eval_function, mode=MODE_AUTO)`` - here, the variable ``de``.
5. Call ``de.start(exit_on_stop=True)``. The `start()` call will block on the
  secondary nodes and call `sys.exit(0)` when the NEAT evolution finishes. This
  means that the following code will only be executed on the primary node.
6. Start the evaluation using ``p.run(de.evaluate, number_of_generations)``.
7. Stop the secondary nodes using ``de.stop()``.
8. You are done. You may want to save the winning genome or show some statistics.

See ``examples/xor/evolve-feedforward-distributed.py`` for a complete example.

Utility functions:

``host_is_local(hostname, port=22)`` returns True if ``hostname`` points to
the local node/host. This can be used to check if a compute node will run as
a primary node or as a secondary node with MODE_AUTO.

``chunked(data, chunksize)``: splits data into a list of chunks with at most
``chunksize`` elements.
"""
from __future__ import print_function

import socket
import select
import struct
import sys
import time
import warnings
import multiprocessing
import base64
import pickle
import json
import threading

try:
    import Queue as queue
except ImportError:
    import queue

# Some of the original code is based on
# http://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing
# According to the website, the code is in the public domain
# ('public domain' links to unlicense.org).
# This means that we can use the code from this website.
# Thanks to Eli Bendersky for making his code open for use.


# modes to determine the role of a compute node
# the primary handles the evolution of the genomes
# the secondary handles the evaluation of the genomes
MODE_AUTO = 0  # auto-determine mode
MODE_PRIMARY = MODE_MASTER = 1  # enforce primary mode
MODE_SECONDARY = MODE_SLAVE = 2  # enforce secondary mode

# states to determine whether the secondaries should shut down
_STATE_RUNNING = 0
_STATE_SHUTDOWN = 1
_STATE_FORCED_SHUTDOWN = 2
_STATE_ERROR = 3


# constants for network communication
_LENGTH_PREFIX = "!Q"
_LENGTH_PREFIX_LENGTH = struct.calcsize(_LENGTH_PREFIX)
_DEFAULT_NETWORK_ENCODING = "utf-8"  # encoding for json messages


class ModeError(RuntimeError):
    """
    An exception raised when a mode-specific method is being
    called without being in the mode - either a primary-specific method
    called by a secondary node or a secondary-specific method called by a primary node.
    """
    pass


class ProtocolError(IOError):
    """
    An Exception raised when either the client or the server does not
    send a valid response.
    """
    pass


class AuthError(Exception):
    """raised if the Authentication failed."""
    pass

def host_is_local(hostname, port=22): # no port specified, just use the ssh port
    """
    Returns True if the hostname points to the localhost, otherwise False.
    """
    hostname = socket.getfqdn(hostname)
    if hostname in ("localhost", "0.0.0.0", "127.0.0.1", "1.0.0.127.in-addr.arpa",
                    "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa"):
        return True
    localhost = socket.gethostname()
    if hostname == localhost:
        return True
    localaddrs = socket.getaddrinfo(localhost, port)
    targetaddrs = socket.getaddrinfo(hostname, port)
    for (ignored_family, ignored_socktype, ignored_proto, ignored_canonname,
         sockaddr) in localaddrs:
        for (ignored_rfamily, ignored_rsocktype, ignored_rproto,
             ignored_rcanonname, rsockaddr) in targetaddrs:
            if rsockaddr[0] == sockaddr[0]:
                return True
    return False


def _determine_mode(addr, mode):
    """
    Returns the mode which should be used.
    If mode is MODE_AUTO, this is determined by checking if 'addr' points to the
    local host. If it does, return MODE_PRIMARY, else return MODE_SECONDARY.
    If mode is either MODE_PRIMARY or MODE_SECONDARY,
    return the 'mode' argument. Otherwise, a ValueError is raised.
    """
    if isinstance(addr, tuple):
        host = addr[0]
    elif type(addr) == type(b"binary_string"):
        host = addr
    else:
        raise TypeError("'addr' needs to be a tuple or an bytestring!")
    if mode == MODE_AUTO:
        if host_is_local(host):
            return MODE_PRIMARY

        return MODE_SECONDARY
    elif mode in (MODE_SECONDARY, MODE_PRIMARY):
        return mode
    else:
        raise ValueError("Invalid mode {!r}!".format(mode))


def chunked(data, chunksize):
    """
    Returns a list of chunks containing at most ``chunksize`` elements of data.
    """
    if chunksize < 1:
        raise ValueError("Chunksize must be at least 1!")
    if int(chunksize) != chunksize:
        raise ValueError("Chunksize needs to be an integer")
    res = []
    cur = []
    for e in data:
        cur.append(e)
        if len(cur) >= chunksize:
            res.append(cur)
            cur = []
    if cur:
        res.append(cur)
    return res


def json_bytes_dumps(obj):
    """
    Encodes obj into json, returning a bytestring.
    This is mainly used for py2/py3 compatibility.
    """
    dumped = json.dumps(obj, ensure_ascii=True)
    encoded = dumped.encode(_DEFAULT_NETWORK_ENCODING)
    return encoded


def json_bytes_loads(bytestr):
    """
    Decodes a bytestring into json.
    This is mainly used for py2/py3 cimpatibility.
    """
    decoded = bytestr.decode(_DEFAULT_NETWORK_ENCODING)
    loaded = json.loads(decoded)
    return loaded


def _serialize_tasks(tasks):
    """serialize a tasklist."""
    # TODO: this needs to be done in a more efficient way.
    return base64.b64encode(pickle.dumps(tasks, -1))


def _load_tasks(s):
    """loads a tasklist from a string returned by _serialize_tasks()"""
    return pickle.loads(base64.b64decode(s))


class _MessageHandler(object):
    """
    Class for managing a socket connection.
    This includes detecting incomplete messages and completing them with
    later messages.
    """
    
    # constants for managing the current state
    _STATE_RECV_PREFIX = 0  # we are currently waiting for the length prefix
    _STATE_RECV_MESSAGE = 1  # we arer currently receiving a message
    
    def __init__(self, s):
        self._s = s
        self._state = self._STATE_RECV_PREFIX
        self._msg_size = _LENGTH_PREFIX_LENGTH
        self._cur_buff = b""
        self.messages = []
    
    def feed(self, data):
        """
        Process received data.
        Returns the number which still need to be received for this message.
        """
        received_a_whole_message = False
        self._cur_buff += data
        while len(self._cur_buff) >= self._msg_size:
            received_a_whole_message = received_a_whole_message or (len(self._cur_buff) >= self._msg_size)
            msg = self._cur_buff[:self._msg_size]
            self._cur_buff = self._cur_buff[self._msg_size:]
            self._handle_message(msg)
        if received_a_whole_message:
            return 0
        else:
            remaining = self._msg_size - len(self._cur_buff)
            return remaining
    
    def _handle_message(self, msg):
        """handle an incomming message as required by self._state"""
        if self._state == self._STATE_RECV_PREFIX:
            self._msg_size = struct.unpack(_LENGTH_PREFIX, msg)[0]
            self._state = self._STATE_RECV_MESSAGE
        elif self._state == self._STATE_RECV_MESSAGE:
            self._msg_size = _LENGTH_PREFIX_LENGTH
            self._state = self._STATE_RECV_PREFIX
            self.messages.append(msg)
        else:
            raise RuntimeError("Internal error: invalid state!")
    
    def send_message(self, msg):
        """sends a message."""
        length = len(msg)
        prefix = struct.pack(_LENGTH_PREFIX, length)
        data = prefix + msg
        self._s.send(data)
    
    def send_json(self, d):
        """serializes d into json, then sends the message."""
        ser = json_bytes_dumps(d)
        return self.send_message(ser)
    
    def recv(self):
        """receives a message from the socket (blocking)."""
        to_recv = 1  # receive one byte initialy
        while True:
            data = self._s.recv(to_recv)
            to_recv = self.feed(data)
            if to_recv == 0:
                return
    
    def get_message(self):
        """if a message was received, return it. Otherwise, receive a message and return it."""
        while len(self.messages) == 0:
            self.recv()
        return self.messages.pop(0)



class DistributedEvaluator(object):
    """An evaluator working across multiple machines"""
    def __init__(
            self,
            addr,
            authkey,
            eval_function,
            secondary_chunksize=1,
            num_workers=None,
            worker_timeout=60,
            mode=MODE_AUTO,
            ):
        """
        ``addr`` should be a tuple of (hostname, port) pointing to the machine
        running the DistributedEvaluator in primary mode. If mode is MODE_AUTO,
        the mode is determined by checking whether the hostname points to this
        host or not.
        ``authkey`` is the password used to restrict access to the manager; see
        ``Authentication Keys`` in the `multiprocessing` manual for more information.
        All DistributedEvaluators need to use the same authkey. Note that this needs
        to be a `str` object for Python 3.X, and should be in 2.7 for compatibility
        (identical in 2.7 to a `str` object).
        ``eval_function`` should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        'secondary_chunksize' specifies the number of genomes that will be sent to
        a secondary at any one time.
        ``num_workers`` is the number of child processes to use if in secondary
        mode. It defaults to None, which means `multiprocessing.cpu_count()`
        is used to determine this value. If 1 in a secondary node, the process creating
        the DistributedEvaluator instance will also do the evaulations.
        ``worker_timeout`` specifies the timeout (in seconds) for a secondary node
        getting the results from a worker subprocess; if None, there is no timeout.
        ``mode`` specifies the mode to run in; it defaults to MODE_AUTO.
        """
        self.addr = addr
        self.authkey = authkey
        self.eval_function = eval_function
        self.secondary_chunksize = secondary_chunksize
        self.slave_chunksize = secondary_chunksize  # backwards compatibility
        if num_workers:
            self.num_workers = num_workers
        else:
            try:
                self.num_workers = max(1,multiprocessing.cpu_count())
            except (RuntimeError, AttributeError): # pragma: no cover
                print("multiprocessing.cpu_count() gave an error; assuming 1",
                      file=sys.stderr)
                self.num_workers = 1
        self.worker_timeout = worker_timeout
        self.mode = _determine_mode(self.addr, mode)
        self.started = False
        self._inqueue = queue.Queue()
        self._outqueue = queue.Queue()
        self._sock_thread = None
        self._va_lock = threading.Lock()

    def __getstate__(self):
        """Required by the pickle protocol."""
        # we do not actually save any state, but we need __getstate__ to be
        # called.
        return True  # return some nonzero value

    def __setstate__(self, state):
        """Called when instances of this class are unpickled."""
        pass

    def is_primary(self):
        """Returns True if the caller is the primary node"""
        return (self.mode == MODE_PRIMARY)

    def is_master(self): # pragma: no cover
        """Returns True if the caller is the primary (master) node"""
        warnings.warn("Use is_primary, not is_master", DeprecationWarning)
        return self.is_primary()

    def start(self, exit_on_stop=True, secondary_wait=0, reconnect=False):
        """
        If the DistributedEvaluator is in primary mode, starts the manager
        process and returns. In this case, the ``exit_on_stop`` argument will
        be ignored.
        If the DistributedEvaluator is in secondary mode, it connects to the manager
        and waits for tasks.
        If in secondary mode and ``exit_on_stop`` is True, sys.exit() will be called
        when the connection is lost.
        ``secondary_wait`` specifies the time (in seconds) to sleep before actually
        starting when in secondary mode.
        If 'reconnect' is True, the secondary nodes will try to reconnect when
        the connection is lost. In this case, sys.exit() will only be called
        when 'exit_on_stop' is True and the primary node send a forced shutdown
        command.
        """
        if self.started:
            raise RuntimeError("DistributedEvaluator already started!")
        self.started = True
        if self.mode == MODE_PRIMARY:
            self._start_primary()
        elif self.mode == MODE_SECONDARY:
            time.sleep(secondary_wait)
            self._start_secondary()
            self._secondary_loop(reconnect=reconnect)
            if exit_on_stop:
                sys.exit(0)
        else:
            raise ValueError("Invalid mode {!r}!".format(self.mode))

    def stop(self, wait=1, shutdown=True, force_secondary_shutdown=False):
        """
        Stops all secondaries.
        'wait' specifies the time (in seconds) to wait before shutting down the
        manager or returning.
        If 'shutdown', shutdown the manager.
        If 'force_secondary_shutdown', shutdown the secondary nodes even if
        they are started with 'reconnect=True'.
        """
        if self.mode != MODE_PRIMARY:
            raise ModeError("Not in primary mode!")
        if not self.started:
            raise RuntimeError("Not yet started!")
        if force_secondary_shutdown:
            self._state = _STATE_FORCED_SHUTDOWN
        else:
            self._state = _STATE_SHUTDOWN
        time.sleep(wait)  # wait is now mostly for backwards compability
        if shutdown:
            try:
                self._listen_s.close()
            except:
                pass
        self.started = False

    def _start_primary(self):
        """Start as the primary"""
        # setup primary specific vars
        self._clients = {}  # socket -> _MessageHandler
        self._s2tasks = {}  # socket -> tasks
        self._authenticated_clients = []  # list of authenticated secondaries
        self._waiting_clients = []  # list of secondaries waiting for tasks
        
        # create socket, bind and listen
        self._bind_and_listen()
        
        # create and start network thread
        self._sock_thread = threading.Thread(
            name="{c} network thread".format(c=self.__class__),
            target=self._primary_sock_thread,
            )
        self._sock_thread.start()
    
    def _bind_and_listen(self):
        """create a socket, bind it and starts listening for connections."""
        self._listen_s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)  # todo: ipv6 support
        self._listen_s.bind(self.addr)
        self._listen_s.listen(3)
    
    def _primary_sock_thread(self):
        """method for the socket thread of the primary node."""
        if self.mode != MODE_PRIMARY:
            raise ModeError("Not a primary node!")
        while self.started:
            to_check_read = [self._listen_s] + list(self._clients.keys())  # list() for python3 compatibility
            to_check_err = [self._listen_s] + list(self._clients.keys())  #  ^^ TODO: is there another way to do this? 
            to_read, to_write, has_err = select.select(to_check_read, [], to_check_err, 0.1)
            if (len(to_read) + len(to_write) + len(has_err)) == 0:
                continue
            
            for s in to_read:
                if s is self._listen_s:
                    # new connection
                    c, addr = s.accept()
                    mh = _MessageHandler(c)
                    self._clients[c] = mh
                else:
                    # data available
                    data = s.recv(4096)  # receive at most 4k bytes. If more are available, recv them in the next iteration.
                    if len(data) == 0:
                        # connection closed.
                        self._remove_client(s)
                        continue
                    mh = self._clients.get(s, None)
                    if mh is None:
                        self._remove_client(s)
                        continue
                    mh.feed(data)
                    
                    while len(mh.messages) > 0:
                        msg = mh.messages.pop(0)
                        try:
                            loaded = json_bytes_loads(msg)
                        except:
                            self._remove_client(s)
                            break
                        action = loaded.get(b"action", None)
                        
                        # authentication
                        if action == b"auth":
                            authkey = loaded.get(b"authkey")
                            if authkey == self.authkey:
                                if s not in self._authenticated_clients:
                                    self._authenticated_clients.append(s)
                                mh.send_json({b"action": "auth_response", b"success": True})
                            else:
                                mh.send_json({b"action": b"auth_response", b"success": False})
                                self._remove_client(s)
                                break
                        elif s not in self._authenticated_clients:
                            # client did not authenticate
                            self._remove_client(s)
                            break
                        
                        # taks distribution
                        elif action == b"get_task":
                            try:
                                tasks = self._inqueue.get(timeout=0)
                            except queue.Empty:
                                self._va_lock.acquire()
                                self._waiting_clients.append(s)
                                self._va_lock.release()
                            else:
                                self._send_tasks(mh, tasks)
                        
                        # results
                        elif action == b"results":
                            results = loaded.get(b"results", None)
                            if results is not None:
                                self._outqueue.put(results)
                                self._va_lock.acquire()
                                if s in self._s2tasks:
                                    del self._s2tasks[s]
                                self._va_lock.release()
                            else:
                                # results not send; reissue tasks if required.
                                self._va_lock.acquire()
                                if s in self._s2tasks:
                                    tasks = self._s2tasks[s]
                                    del self._s2tasks[s]
                                self._va_lock.release()
                                self._inqueue.put(tasks)
                        
                        else:
                            # unknown message; this is probably an error.
                            self._remove_client(s)
                            break
            
            # for s in to_write:
            #    # not required
            #    pass
            
            for s in has_err:
                if s is self._listen_s:
                    # cant listen anymore
                    # try to rebind
                    try:
                        s.close()
                    except:
                        # ignore exception during socket close,
                        # socket may already be closed.
                        pass
                    if self._state == self._STATE_RUNNING:
                        try:
                            self._bind_and_listen()
                        except:
                            self._state = self._STATE_ERROR
                            break
                    else:
                        # server stopped or not yet started,
                        # do not rebind and break this loop instead.
                        break
                else:
                    self._remove_client(s)
        
        # stop connected clients
        forced = (self._state == _STATE_FORCED_SHUTDOWN)
        for s in self._clients:
            self._send_stop(s, forced=forced)
            self._remove_client(s)
    
    def _remove_client(self, s):
        """closes and removes the client."""
        self._va_lock.acquire()
        try:
            s.close()
        except:
            pass
        if s in self._clients:
            del self._clients[s]
        if s in self._authenticated_clients:
            self._authenticated_clients.remove(s)
        if s in self._s2tasks:
            tasks = self._s2tasks[s]
            del self._s2tasks[s]
        else:
            tasks = None
        self._va_lock.release()
        if tasks is not None:
            self._add_tasks(tasks)
    
    def _send_tasks(self, mh, tasks):
        """sends some tasks to a secondary connected through the message handler mh."""
        ser_tasks = _serialize_tasks(tasks)
        mh.send_json(
               {
                   "action": "tasks",
                   "tasks": ser_tasks,
               },
           )
    
    def _send_stop(self, s, forced=False):
        """sends a stop message to a client."""
        self._va_lock.acquire()
        mh = self._clients.get(s, _MessageHandler(s))
        mh.send_json(
                {
                    "action": "stop",
                    "forced": forced,
                }
            )
        self._va_lock.release()
     
    def _add_tasks(self, tasks):
        """adds a task for evaluation."""
        if len(self._waiting_clients) > 0:
            self._va_lock.acquire()
            s = self._waiting_clients.pop(0)
            mh = self._clients.get(s, None)
            self._va_lock.release()
            if mh is None:
                self._remove_client(s)
                return self._add_tasks(tasks)
            self._send_tasks(mh, tasks)
        else:
            self._inqueue.put(tasks)

    def _start_secondary(self):
        """Start as a secondary."""
        # TODO: implement this
        pass
    
    def _reset(self):
        """resets the internal state of the secondary nodes."""
        # connect
        self._s = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        self._s.connect(self.addr)
        
        # create a _MessageHandler
        self._mh = _MessageHandler(self._s)
        
        # auth
        self._mh.send_json(
                {
                    "action": "auth",
                    "authkey": self.authkey,
                }
            )
        response = json_bytes_loads(self._mh.get_message())
        if response.get("action", None) != "auth_response":
            self._s.close()
            raise ProtocolError("Server did not send an auth response!")
        success = response.get("success", False)
        if not success:
            self._s.close()
            raise AuthError("Invalid authkey!")

    def _secondary_loop(self, reconnect=False):
        """The worker loop for the secondary nodes."""
        if self.num_workers > 1:
            pool = multiprocessing.Pool(self.num_workers)
        else:
            pool = None
        self._should_reconnect = True
        while self._should_reconnect:
            running = True
            try:
                self._reset()
            except (socket.error, EOFError, IOError, OSError, socket.gaierror,):
                continue
            while running:
                try:
                    tasks = self._get_tasks()
                    if tasks is None:
                        break
                except (socket.error, EOFError, IOError, OSError, socket.gaierror):
                    break
                if pool is None:
                    res = []
                    for genome_id, genome, config in tasks:
                        fitness = self.eval_function(genome, config)
                        res.append((genome_id, fitness))
                else:
                    genome_ids = []
                    jobs = []
                    for genome_id, genome, config in tasks:
                        genome_ids.append(genome_id)
                        jobs.append(
                            pool.apply_async(
                                self.eval_function, (genome, config)
                                )
                            )
                    results = [
                        job.get(timeout=self.worker_timeout) for job in jobs
                        ]
                    res = zip(genome_ids, results)
                try:
                    self._send_results(res)
                except (socket.error, EOFError, IOError, OSError, socket.gaierror):
                    break

            if not reconnect:
                self._should_reconnect = False
                break
        try:
            self._s.close()
        except:
            pass
        if pool is not None:
            pool.terminate()
    
    def _get_tasks(self):
        """
        Receives some tasks from the primary.
        This method returns either the received tasks or None if the
        secondary was stopped by the primary.
        """
        tosend = {"action": "get_task"}
        self._mh.send_json(tosend)
        while True:
            msg = json_bytes_loads(self._mh.get_message())
            action = msg.get("action", None)
            
            if action == "stop":
                forced = msg.get("forced", False)
                if forced:
                    self._should_reconnect = False
                return None
                
            elif action == "tasks":
                tasks = msg.get("tasks", None)
                if tasks is not None:
                    return _load_tasks(tasks)
    
    def _send_results(self, results):
        """sends the results to the primary node."""
        self._mh.send_json(
                {
                    "action": "results",
                    "results": results,
                }
            )

    def evaluate(self, genomes, config):
        """
        Evaluates the genomes.
        This method raises a ModeError if the
        DistributedEvaluator is not in primary mode.
        """
        if self.mode != MODE_PRIMARY:
            raise ModeError("Not in primary mode!")
        tasks = [(genome_id, genome, config) for genome_id, genome in genomes]
        id2genome = {genome_id: genome for genome_id, genome in genomes}
        tasks = chunked(tasks, self.secondary_chunksize)
        n_tasks = len(tasks)
        for tasklist in tasks:
            self._add_tasks(tasklist)
        tresults = []
        while len(tresults) < n_tasks:
            try:
                sr = self._outqueue.get(block=True, timeout=0.2)
            except (queue.Empty):
                continue
            tresults.append(sr)
        results = []
        for sr in tresults:
            results += sr
        for genome_id, fitness in results:
            genome = id2genome[genome_id]
            genome.fitness = fitness
