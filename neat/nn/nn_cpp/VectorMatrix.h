// *************************************//
// Classes for vectors and matrices     // 
// Author: Randall D. Beer              //
//         mypage.iu.edu/~rdbeer/       //
// *************************************//

#ifndef _VectorMatrix_H_
#define _VectorMatrix_H_

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdarg>

using namespace std;

#define DEBUG 0


// *******
// TVector
// *******

// The TVector class declaration

template<class EltType>
class TVector {
    public:
        // Constructors
        TVector(void);
        TVector(int LowerBound, int UpperBound);
        TVector(TVector<EltType> &v);
        // The destructor
        ~TVector();
        // Accessors
        int Size(void) {return ub - lb + 1;};
        void SetSize(int NewSize) {SetBounds(lb,NewSize+lb-1);};
        int LowerBound(void) {return lb;};
        void SetLowerBound(int NewLB) {SetBounds(NewLB,ub);};
        int UpperBound(void) {return ub;};
        void SetUpperBound(int NewUB) {SetBounds(lb,NewUB);};
        void SetBounds(int NewLB, int NewUB);
        // Other stuff
        void FillContents(EltType value);
        void InitializeContents(EltType v1,...);
        // Vector i/o
        void BinaryWriteVector(ofstream& bofs);
        void BinaryReadVector(ifstream& binfs);
        // Overloaded operators
        EltType &operator[](int index) 
        {
        #if !DEBUG
            return Vector[index];
        #else
            return (*this)(index);
        #endif
        };
        inline EltType &operator()(int index);
        inline TVector<EltType> &operator=(TVector<EltType> &v);
        
    protected:
        int lb, ub;
        EltType *Vector;
};


// The default constructor

template<class EltType>
TVector<EltType>::TVector(void) 
{
    lb = 1; ub = 0;
}


// The standard constructor

template<class EltType>
TVector<EltType>::TVector(int LB, int UB) 
{
    lb = 1; ub = 0; 
    SetBounds(LB,UB);
}


// The copy constructor

template<class EltType>
TVector<EltType>::TVector(TVector<EltType> &v)
{
    lb = 1; ub = 0; 
    SetBounds(v.LowerBound(),v.UpperBound());
    for (int i = lb; i <= ub; i++)
        Vector[i] = v[i];
}


// The destructor

template<class EltType>
TVector<EltType>::~TVector(void)
{
    SetSize(0);
}


// Set the bounds of a TVector, reallocating space as necessary and
// preserving as much of the previous contents as possible

template<class EltType>
void TVector<EltType>::SetBounds(int newlb, int newub)
{
    // Only do it if we have to
    if (lb == newlb && ub == newub) return;
    // Save the old info and init the new
    EltType *OldVector = Vector;
    int oldlb = lb, oldub = ub, oldlen = ub - lb + 1, len = newub - newlb + 1;
    lb = newlb; ub = newub;
    // No negative length vectors allowed!
    if (len < 0) {
        cerr << "Attempt to allocate a negative length TVector\n";
        exit(0);
    }
    // Allocate the new storage and copy as much of the old info as possible
    if (len != 0) {
        Vector = new EltType[len] - lb;
        if (oldlen != 0)
            for (int i = oldlb, j = lb; i <= oldub && j <= ub; i++,j++)
                Vector[j] = OldVector[i];
    }
    // Recover the old storage
    if (oldlen != 0) delete [] (OldVector + oldlb);
}


// Fill a TVector with the given value

template<class EltType>
void TVector<EltType>::FillContents(EltType value)
{
    for (int i = lb; i <= ub; i++)
        Vector[i] = value;
}


// Initialize a TVector with given contents

template<class EltType>
void TVector<EltType>::InitializeContents(EltType v1,...)
{
    va_list ap;
    
    if (Size() == 0) return;
    Vector[lb] = v1;
    va_start(ap,v1);
    for (int i = lb+1; i <= ub; i++)
        Vector[i] = va_arg(ap,EltType);
    va_end(ap);
}
        

// Overload the () operator to provide safe indexing

template<class EltType>
inline EltType &TVector<EltType>::operator()(int index)
{
    if (index < lb || index > ub) 
    {
        cerr << "Vector index " << index << " out of bounds\n";
         exit(0);
    }
    return Vector[index];
}


// Overload the = operator to copy one TVector to another

template<class EltType>
inline TVector<EltType> &TVector<EltType>::operator=(TVector<EltType> &v)
{
    SetBounds(v.LowerBound(),v.UpperBound());
    for (int i = lb; i <= ub; i++)
        Vector[i] = v[i];
    return *this;
}
        
        
// Overload the stream insertion operator to recognize a TVector

template<class EltType>
ostream& operator<<(ostream& os, TVector<EltType>& v)
{
    for (int i = v.LowerBound(); i < v.UpperBound(); i++)
        os << v[i] << " ";
    if (v.Size() > 0) os << v[v.UpperBound()];
    return os;
}


// Write and read a TVector in binary
// (Thanks to Chad Seys)

template<class EltType>
void TVector<EltType>::BinaryWriteVector(ofstream& bofs)
{
    int thisSize = ub-lb+1;
    bofs.write((const char*) &(lb), sizeof(lb));
    bofs.write((const char*) &(ub), sizeof(ub));
    
    for (int i = lb; i < ub; i++) {
        bofs.write((const char *) &(Vector[i]), sizeof(Vector[i]));
    }    
    if (thisSize > 0) {
        bofs.write((const char *) &(Vector[ub]), sizeof(Vector[ub]));
    }
}

template<class EltType>
void TVector<EltType>::BinaryReadVector(ifstream& bifs)
{
    int LB;
    int UB;
    bifs.read((char *) &(LB), sizeof(LB));
    bifs.read((char *) &(UB), sizeof(UB));
    SetLowerBound(LB);
    SetUpperBound(UB);
    
    for (int i = LB; i <= UB; ++i) {
        bifs.read((char *) &(Vector[i]),sizeof(Vector[i]));
    }
}


// *******
// TMatrix
// *******

// The TMatrix class declaration

template<class EltType>
class TMatrix {
    public:
        // Constructors
        TMatrix(void);
        TMatrix(int RowLowerBound, int RowUpperBound, int ColumnLowerBound, int ColumnUpperBound);
        TMatrix(TMatrix<EltType> &m);
        // The destructor
        ~TMatrix();
        
        // Accessors
        int RowSize(void) {return rowlen;};
        void SetRowSize(int NewSize) {SetBounds(lb1,lb1+NewSize-1,lb2,ub2);};
        int ColumnSize(void) {return collen;};
        void SetColumnSize(int NewSize) {SetBounds(lb1,ub1,lb2,lb2+NewSize-1);};
        void SetSize(int NewRowSize,int NewColSize) 
            {SetBounds(lb1,lb1+NewRowSize-1,lb2,lb2+NewColSize-1);};
        int RowLowerBound(void) {return lb1;};
        void SetRowLowerBound(int newlb1) {SetBounds(newlb1,ub1,lb2,ub2);};
        int RowUpperBound(void) {return ub1;};
        void SetRowUpperBound(int newub1) {SetBounds(lb1,newub1,lb2,ub2);};
        int ColumnLowerBound(void) {return lb2;};
        void SetColumnLowerBound(int newlb2) {SetBounds(lb1,ub1,newlb2,ub2);};
        int ColumnUpperBound(void) {return ub2;};
        void SetColumnUpperBound(int newub2) {SetBounds(lb1,ub1,lb2,newub2);};
        void SetBounds(int newlb1, int newub1, int newlb2, int newub2);
        // Overloaded operators
        EltType* operator[](int index) 
        {
        #if !DEBUG
            return Matrix[index];
        #else
                if (index < lb1 || index > ub1)
                {
                    cerr << "Matrix index " << index << " out of bounds\n";
                    exit(0);
                }
                return Matrix[index];
        #endif
        };
        inline EltType &operator()(int i,int j);
        inline TMatrix<EltType> &operator=(TMatrix<EltType> &m);
        // Other stuff
        void FillContents(EltType x);
        void InitializeContents(EltType v1,...);
        
    protected:
        int lb1, ub1, lb2, ub2, collen, rowlen;
        EltType **Matrix;
};


// The default constructor

template<class EltType>
TMatrix<EltType>::TMatrix(void)
{
    lb1 = lb2 = 1; ub1 = ub2 = 0; collen = 0; rowlen = 0;
}


// The standard constructor

template<class EltType>
TMatrix<EltType>::TMatrix(int RowLowerBound, int RowUpperBound, 
                          int ColumnLowerBound, int ColumnUpperBound)
{
    lb1 = lb2 = 1; ub1 = ub2 = 0; collen = 0; rowlen = 0;
    SetBounds(RowLowerBound,RowUpperBound,ColumnLowerBound,ColumnUpperBound);
}


// The copy constructor

template<class EltType>
TMatrix<EltType>::TMatrix(TMatrix<EltType> &m)
{
    lb1 = lb2 = 1; ub1 = ub2 = 0; collen = 0; rowlen = 0;
    SetBounds(m.RowLowerBound(),m.RowUpperBound(),m.ColumnLowerBound(),m.ColumnUpperBound());
    for (int i = lb1; i <= ub1; i++)
        for (int j = lb2; j <= ub2; j++)
            Matrix[i][j] = m[i][j];
}


// The destructor

template<class EltType>
TMatrix<EltType>::~TMatrix()
{
    SetSize(0,0);
}


// Set the bounds of a TMatrix to the given values, reallocating space as necessary.
// Note that, unlike for TVectors, we do not try to preserve the previous contents.

template<class EltType>
void TMatrix<EltType>::SetBounds(int newlb1, int newub1, int newlb2, int newub2)
{
    // Only do it if we have to
    if (newlb1 == lb1 && newub1 == ub1 && newlb2 == lb2 && newub2 == ub2) return;
    // If storage is currently allocated, reclaim it
    if (collen != 0) {
        if (rowlen != 0)
            for (int i = lb1; i <= ub1; i++)
                delete (Matrix[i] + lb2);
        delete [] (Matrix + lb1);
    }
    // Save the new bounds info
    lb1 = newlb1; ub1 = newub1; lb2 = newlb2; ub2 = newub2; 
    collen = ub1 - lb1 + 1; rowlen = ub2 - lb2 + 1;
    // No negative sizes allowed!
    if (collen < 0 || rowlen < 0) {
        cerr << "Attempt to allocate a negative sized TMatrix\n";
        exit(0);
    }
    // If new storage is needed, allocate it
    if (collen != 0) {
        Matrix = new EltType*[collen] - lb1;
        if (rowlen != 0)
                for (int i = lb1; i <= ub1; i++)
                    Matrix[i] = new EltType[rowlen] - lb2;
        else
            for (int i = lb1; i <= ub1; i++)
                Matrix[i] = NULL;
    }
}


// Fill a TMatrix with the given value

template<class EltType>
void TMatrix<EltType>::FillContents(EltType x)
{
    for (int i = lb1; i <= ub1; i++)
        for (int j = lb2; j <= ub2; j++)
            Matrix[i][j] = x;
}


// Initialize a TMatrix with given contents

template<class EltType>
void TMatrix<EltType>::InitializeContents(EltType v1,...)
{
    va_list ap;
    
    if (rowlen == 0 || collen == 0) return;
    Matrix[lb1][lb2] = v1;
    va_start(ap,v1);
    for (int j = lb2+1; j <= ub2; j++)
        Matrix[lb1][j] = va_arg(ap,EltType);
    for (int i = lb1+1; i <= ub1; i++)
        for (int j = lb2; j <= ub2; j++)
            Matrix[i][j] = va_arg(ap,EltType);
    va_end(ap);
}


// Overload the () operator to provide safe indexing

template<class EltType>
inline EltType &TMatrix<EltType>::operator()(int i,int j)
{
    if (i < lb1 || i > ub1 || j < lb2 || j > ub2)
    {
        cerr << "Matrix indices (" << i << "," << j << ") out of bounds\n";
        exit(0);
    }
    return Matrix[i][j];
}


// Overload the = operator to copy one TMatrix to another

template<class EltType>
inline TMatrix<EltType> &TMatrix<EltType>::operator=(TMatrix<EltType> &m)
{
    SetBounds(m.RowLowerBound(),m.RowUpperBound(),m.ColumnLowerBound(),m.ColumnUpperBound());
    for (int i = lb1; i <= ub1; i++)
        for (int j = lb2; j <= ub2; j++)
            Matrix[i][j] = m[i][j];
    return *this;
}
        
        
// Overload the stream insertion operator to recognize a TMatrix

template<class EltType>
ostream& operator<<(ostream& os, TMatrix<EltType> &m)
{
    int i,j;
    
    for (i = m.RowLowerBound(); i < m.RowUpperBound(); i++) {
        for (j = m.ColumnLowerBound(); j < m.ColumnUpperBound(); j++) 
            os << m[i][j] << " ";
        if (m.ColumnSize() > 0) os << m[i][m.ColumnUpperBound()] << endl;
    }
    if (m.RowSize() > 0) {
        for (j = m.ColumnLowerBound(); j < m.ColumnUpperBound(); j++)
            os << m[m.RowUpperBound()][j] << " ";
        if (m.ColumnSize() > 0) os << m[m.RowUpperBound()][m.ColumnUpperBound()];
    }
    return os;
}
#endif
