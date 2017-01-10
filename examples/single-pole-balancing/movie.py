import gizeh as gz
import moviepy.editor as mpy
from cart_pole import CartPole


def make_movie(net, force_function, duration_seconds, output_filename):
    w, h = 300, 100
    scale = 300 / 6

    cart = gz.rectangle(scale * 0.5, scale * 0.25, xy=(150, 80), stroke_width=1, fill=(0, 1, 0))
    pole = gz.rectangle(scale * 0.1, scale * 1.0, xy=(150, 55), stroke_width=1, fill=(1, 1, 0))

    sim = CartPole()

    def make_frame(t):
        inputs = sim.get_scaled_state()
        if hasattr(net, 'activate'):
            action = net.activate(inputs)
        else:
            action = net.advance(inputs, sim.time_step, sim.time_step)

        sim.step(force_function(action))

        surface = gz.Surface(w, h, bg_color=(1, 1, 1))

        # Convert position to display units
        visX = scale * sim.x

        # Draw cart.
        group = gz.Group((cart,)).translate((visX, 0))
        group.draw(surface)

        # Draw pole.
        group = gz.Group((pole,)).translate((visX, 0)).rotate(sim.theta, center=(150 + visX, 80))
        group.draw(surface)

        return surface.get_npimage()

    clip = mpy.VideoClip(make_frame, duration=duration_seconds)
    clip.write_videofile(output_filename, codec="mpeg4", fps=50)
