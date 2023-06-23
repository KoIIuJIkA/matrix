
#include <iostream>
#include <initializer_list>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

namespace linalg {
    // struct TextMode;
    class LinAlgError : public std::exception {
        char* message;
    public:
        LinAlgError(char* message) {
            this->message = message;
        }
        const char* what() const noexcept override {
            return message;
        }
    };

    template<typename T = double>
    class Matrix {
        T **m_ptr = nullptr;
        int m_rows, m_columns;
        void clearMemory(int** a, int n) {
            for (int i = 0; i < n; i++) {
                delete[] a[i];
            }
            delete [] a;        
        }
        
        void swap_rows(int i, int j) {
            for (int k = 0; k < m_columns; k++)
                std::swap(m_ptr[i * m_columns + k], m_ptr[j * m_columns + k]);
        }
    public:
        template<typename T_new> friend
        class Matrix;
        ///Default constructor.
        Matrix(int rows = 1, int columns = 1) noexcept                  
            : m_rows(rows), m_columns(columns) {
            m_ptr = new T*[m_rows];
            for(int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
        }
        ///Iterator constructor.
        template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
        Matrix(std::initializer_list<T_new> list) noexcept      
            : m_rows(1), m_columns(list.size()) {
            auto iter = list.begin();
            m_ptr = new T*[1];
            for(int i = 0; i < 1; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for(int i = 0; i < 1; ++i) {
                for(int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<T>(*(iter + j));
                }
            }
        }
        ///Iterator constructor.
        template<typename T_new>
        Matrix(std::initializer_list<std::initializer_list<T_new>> list) noexcept {
            auto iter = list.begin();
            m_rows = list.size();
            m_columns = iter->size();
            m_ptr = new T *[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<T>(*((iter + i)->begin() + j));
                }
            }
        }
        ///Copy constructor.
        template<typename T_new>
        Matrix(const Matrix<T_new> &other) noexcept
            : m_rows(other.m_rows), m_columns(other.m_columns) {
            m_ptr = new T*[m_rows];
            for(int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for(int i = 0; i < m_rows; ++i) {
                for(int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<T>(other.m_ptr[i][j]);
                }
            }
        }
        ///Copy constructor.
        Matrix(const Matrix &other) noexcept { 
            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T*[m_rows];
            for(int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for(int i = 0; i < m_rows; ++i) {
                for(int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = other.m_ptr[i][j];
                }
            }
        }
        ///Move constructor.
        template<typename T_new>                
        Matrix(Matrix<T_new> &&other) noexcept {
            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T *[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<T>(other.m_ptr[i][j]);
                }
                other.m_ptr[i] = nullptr;
            }
            other.m_ptr = nullptr;
            other.m_rows = 0;
            other.m_columns = 0;
        }
        ///Move constructor.
        Matrix(Matrix &&other) noexcept {                      
            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T *[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = other.m_ptr[i][j];
                }
                other.m_ptr[i] = nullptr;
            }
            other.m_ptr = nullptr;
            other.m_rows = 0;
            other.m_columns = 0;
        }
        ///Сopy assignment operator.
        template<typename T_new>
        Matrix &operator=(const Matrix<T_new> &other) noexcept {
            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T *[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for (int i = 0; i < other.m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<T>(other.m_ptr[i][j]);
                }
            }
            return *this;
        }
        ///Сopy assignment operator.
        Matrix &operator=(const Matrix &other) noexcept {  
            if (&other == this) {
                return *this;
            }
            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T *[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for (int i = 0; i < other.m_rows; ++i)
            {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = other.m_ptr[i][j];
                }
            }
            return *this;
        }
        ///Move assignment operator.
        template<typename T_new>
        Matrix &operator=(Matrix<T_new> &&other) noexcept {    
            if (&other == this) {
                return *this;
            }
            for (int i = 0; i < m_rows; ++i) {
                delete[] m_ptr[i];
            }
            delete[] m_ptr;
            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T *[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<T>(other.m_ptr[i][j]);
                }
                other.m_ptr[i] = nullptr;
            }
            other.m_ptr = nullptr;
            other.m_rows = 0;
            other.m_columns = 0;
            return *this;
        }
        ///Move assignment operator.
        Matrix &operator=(Matrix &&other) noexcept {   
            if (&other == this) {
                return *this;
            }
            for (int i = 0; i < m_rows; ++i) {
                delete[] m_ptr[i];
            }
            delete[] m_ptr;

            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_ptr = new T*[m_rows];
            for (int i = 0; i < m_rows; ++i) {
                m_ptr[i] = new T[m_columns];
            }
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = other.m_ptr[i][j];
                }
                other.m_ptr[i] = nullptr;
            }
            other.m_ptr = nullptr;
            other.m_rows = 0;
            other.m_columns = 0;
            return *this;
        }
        ///Ostream override.
        friend std::ostream &operator<<(std::ostream &os, const Matrix &other) noexcept {  
            os << std::endl;
            for (int i = 0; i < other.m_rows; ++i) {
                for (int j = 0; j < other.m_columns; ++j) {                  
                    std::cout << (j == 0 ? "|" : "") << other.m_ptr[i][j];
                    if (j + 1 == other.m_columns) {
                        std::cout << "|" << std::endl;
                    }
                    else
                        std::cout << std::setw(3);
                }
            }
            os << std::endl;
            return os;
        }
        ///Function call operator.
        T &operator()(int row, int col) {
            try {
                if (row > m_rows - 1 || col > m_columns - 1) {
                    throw (std::logic_error("no such index in this matrix"));
                }
            }
            catch (std::logic_error &exception) { 
                std::cout << exception.what() << std::endl;
            }
            return *(*(m_ptr + row) + col);
        }
        ///Function call operator.
        const T &operator()(int row, int col) const {
            try {
                if (row > m_rows - 1 || col > m_columns - 1) {
                    throw (std::logic_error("no such index in this matrix"));
                }
            }
            catch (std::logic_error &exception) {
                std::cout << exception.what() << std::endl;
            }
            return *(*(m_ptr + row) + col);
        }
        template<typename T_new>                   
        Matrix &operator+=(Matrix<T_new> m2) noexcept {
            if (m_rows == m2.m_rows && m_columns == m2.m_columns) {
                for (int i = 0; i < m_rows; ++i) {
                    for (int j = 0; j < m_columns; ++j) {
                        m_ptr[i][j] = m_ptr[i][j] + static_cast<T>(m2(i, j));
                    }
                }
                return *this;
            } 
            else  {
                std::cerr << "matrices of different sizes\n";
                return *this;
            }
        }
        template<typename T_new>
        Matrix &operator-=(Matrix<T_new> m2) noexcept {
            if (m_rows == m2.m_rows && m_columns == m2.m_columns) {
                for (int i = 0; i < m_rows; ++i) {
                    for (int j = 0; j < m_columns; ++j) {
                        m_ptr[i][j] = m_ptr[i][j] - static_cast<T>(m2(i, j));
                    }
                }
                return *this;
            } 
            else {
                std::cerr << "matrices of different sizes\n";
                return *this;
            }
        }
        Matrix &operator*=(const int b) noexcept {
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = m_ptr[i][j] * b;
                }
            }
            return *this;
        }
        Matrix &operator*=(const double b) noexcept {
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    m_ptr[i][j] = static_cast<double>(m_ptr[i][j] * b);
                }
            }
            return *this;
        }
        template<typename T_new>
        friend Matrix operator+(Matrix m1, Matrix<T_new> m2) noexcept {
            if (m1.m_rows == m2.m_rows && m1.m_columns == m2.m_columns) {
                Matrix<T> tmp(m1);
                for (int i = 0; i < m1.m_rows; ++i) {
                    for (int j = 0; j < m1.m_columns; ++j) {
                        tmp(i, j) = tmp(i, j) + static_cast<T>(m2(i, j));
                    }
                }
                return tmp;
            } 
            else {
                std::cerr << "matrices of different sizes\n";
                return 0;
            }
        }
        template<typename T_new>
        friend Matrix operator-(Matrix m1, Matrix<T_new> m2) noexcept {
            if (m1.m_rows == m2.m_rows && m1.m_columns == m2.m_columns) {
                Matrix<T> tmp(m1);
                for (int i = 0; i < m1.m_rows; ++i) {
                    for (int j = 0; j < m1.m_columns; ++j) {
                        tmp(i, j) = tmp(i, j) - static_cast<T>(m2(i, j));
                    }
                }
                return tmp;
            } 
            else {
                std::cerr << "matrices of different sizes\n";
                return 0;
            }
        }
        template<typename T_new>
        friend Matrix operator*(Matrix m1, T_new b) noexcept {
            Matrix<T> tmp(m1);
            for (int i = 0; i < m1.m_rows; ++i) {
                for (int j = 0; j < m1.m_columns; ++j) {
                    tmp(i, j) = tmp(i, j) * static_cast<T>(b);
                }
            }
            return tmp;
        }
        template<typename T_new>
        friend Matrix operator/(Matrix<T> m1, T_new b) noexcept {

            Matrix<T> tmp(m1);
            for (int i = 0; i < m1.m_rows; i++) {
                for (int j = 0; j < m1.m_columns; j++) {
                    tmp(i, j) = (tmp(i, j) / b);
                }
            }
            return tmp;
        }
        template<typename T_new>
        friend Matrix operator*(Matrix m1, Matrix<T_new> m2) noexcept {
            if (m1.m_columns == m2.m_rows) {
                Matrix<T> tmp(m1.m_rows, m2.m_columns);
                for (int i = 0; i < m1.m_rows; ++i) {
                    for (int j = 0; j < m2.m_columns; ++j) {
                        for (int k = 0; k < m1.m_columns; k++) {
                            tmp(i, j) = tmp(i, j) + static_cast<T>(m1(i, k) * m2(k, j));
                        }
                    }
                }
                return tmp;
            } 
            else {
                std::cerr << "first_columns != second_rows\n";
                return 0;
            }
        }

        template<typename T_new>
        bool operator==(Matrix<T_new> m2) {
            int flag = 1;
            if (m_rows == m2.m_rows && m_columns == m2.m_columns) {
                for (int i = 0; i < m_rows; i++) {
                    for (int j = 0; j < m_columns; j++) {
                       if(m_ptr[i][j]!= static_cast<T>(m2(i, j))){flag = 0;};

                    }
                }
                return (flag==1);
            } else {
                throw (LinAlgError(const_cast<char*>("Different sized matrices cannot be equal")));
            }
        }


        template<typename T_new>
        bool operator!=(Matrix<T_new> &other) {
            return !(other == *this);
        }

        ~Matrix() {
            for (int i = 0; i < m_rows; ++i) {
                delete[] m_ptr[i];
            }
            delete[] m_ptr;
        }

        T det() const {
            if (m_rows != m_columns) {
                throw LinAlgError(const_cast<char*>("Cannot calculate determinant of non-square matrix."));
            }
            if (m_rows == 1) {
                return m_ptr[0][0];
            }
            T det = 0;
            for (int i = 0; i < m_columns; i++) {
                Matrix submatrix(m_rows - 1, m_columns - 1);
                for (int j = 1; j < m_rows; j++) {
                    for (int k = 0; k < m_columns; k++) {
                        if (k != i) {
                            submatrix(j - 1, k < i ? k : k - 1) = (*this)(j, k);
                        }
                    }
                }
                T subdet = submatrix.det();
                det += ((i % 2 == 0) ? 1 : -1) * m_ptr[0][i] * subdet;
            }
            return det;
        }

        int rank() {
            Matrix<T> tmp = *this;
            int rank = m_columns;
            for (int row = 0; row < rank; row++) {
                if (tmp.m_ptr[row][row] != 0) {
                    for (int col = 0; col < m_rows; col++) {
                        if (col != row) {
                            double mult = tmp.m_ptr[col][row] / tmp.m_ptr[row][row];
                            for (int i = row; i < rank; i++)
                                tmp.m_ptr[col][i] -= mult * tmp.m_ptr[row][i];
                        }
                    }
                } else {
                    bool reduce = true;
                    for (int i = row + 1; i < m_rows; i++) {
                        if (tmp.m_ptr[i][row] != 0) {
                            std::swap(tmp.m_ptr[row], tmp.m_ptr[i]);
                            reduce = false;
                            break;
                        }
                    }
                    if (reduce) {
                        rank--;
                        for (int i = 0; i < m_rows; i++)
                            tmp.m_ptr[i][row] = tmp.m_ptr[i][rank];
                    }
                    row--;
                }
            }
            return rank;
        }
        double norm() {
            double norm = 0;
            for (int i = 0; i < m_rows; ++i) {
                for (int j = 0; j < m_columns; ++j) {
                    norm += m_ptr[i][j] * m_ptr[i][j];
                }
            }
            return std::sqrt(norm);
        }
        double trace() {
            double trace = 0;
            if (m_rows != m_columns) {
                throw LinAlgError(const_cast<char*>("m_rows != m_columns"));
            }
            for (int i = 0; i < m_rows; ++i) {
                trace += m_ptr[i][i];
            }
            return trace;
        }
        friend Matrix<T> transpose(Matrix<T> &other) {
            Matrix<T> tmp(other);
            if (tmp.m_rows != tmp.m_columns) {
                std::cerr << "m_rows != m_columns" << std::endl;
                return 0;
            }
            for (int i = 0; i < tmp.m_rows; ++i) {
                for (int j = i; j < tmp.m_columns; ++j) {
                    std::swap(tmp.m_ptr[i][j], tmp.m_ptr[j][i]);
                }
            }
            return tmp;
        }
        friend Matrix<T> inv( Matrix<T> &second) {
            Matrix<T> tmp = second;
            if (tmp.det() == 0) {
                throw LinAlgError(const_cast<char *>("the matrix has no inverse"));
            }
            for (int i = 0; i < tmp.m_rows; ++i) {
                for (int j = 0; j < tmp.m_columns; ++j) {
                    tmp(i, j) = static_cast<double>(pow(-1, i + j)) * (tmp.minor(second, i, j)).det();
                }
            }
            return transpose(tmp)* (1 / second.det()) ;
        }

        friend Matrix<T> pow(Matrix<T> &matrix, int p) {
            Matrix<T> new_matrix(matrix);
            for (int i = 1; i < p; i++) {
                new_matrix = new_matrix * matrix;
            }
		    return new_matrix;
	    }

        Matrix<T> minor(Matrix<T> &other, int r, int c) const {
            int t_row = 0, t_column = 0;
            Matrix<T> minor(other.m_rows - 1, other.m_columns - 1);
            for (int i = 0; i < other.m_rows; i++) {
                for (int j = 0; j < other.m_columns; j++) {
                    if (i == r) {
                        --t_row;
                        break;
                    }
                    else if (j != c) {
                        minor.m_ptr[t_row][t_column] = other.m_ptr[i][j];
                        ++t_column;
                    }
                }
                ++t_row;
                t_column = 0;
            }
            return minor;
	    }
        
        friend double angle(Matrix<T> first, Matrix<T> second) {
            return std::acos(multy_scalar(first, second) / (first.norm() * second.norm()));
        }
        friend double multy_scalar(Matrix<T> first, Matrix<T> second) {
            if (first.m_columns != second.m_columns && first.m_rows != second.m_rows) {
                throw LinAlgError(const_cast<char*>("unable to find angle"));
            }
            double product = 0;
            for (int i = 0; i < first.m_rows; ++i) {
                for (int j = 0; j < first.m_columns; ++j) {
                    product += first.m_ptr[i][j] * second.m_ptr[i][j];
                }
            }
            return product;
        }
        friend Matrix<T> multy_vector(Matrix<T> first, Matrix<T> second) {
            if (first.m_columns != 3 || second.m_columns != 3 || first.m_rows != 1 || second.m_rows != 1 ) {
                throw LinAlgError(const_cast<char*>("bad shape."));
		    }
            T x = first(0, 1) * second(0, 2) - first(0, 2) * second(0, 1);
            T y = first(0, 2) * second(0, 0) - first(0, 0) * second(0, 2);
            T z = first(0, 0) * second(0, 1) - first(0, 1) * second(0, 0);
            Matrix<T> new_vec( { x, y, z } );
            return new_vec;
        }
        friend Matrix<double> unit(Matrix<T> &vec) {
            if (vec.m_rows != 1) {
                throw LinAlgError(const_cast<char*>("bad shape."));
            }
            Matrix<double> new_vec(vec);
            double len = vec.norm();
            for (int i = 0; i < vec.m_columns; i++) {
                new_vec(0, i) /= len;
            }
            return new_vec;
        }
        friend Matrix<double> solve(Matrix<T>& matrix_a, Matrix<T>& vec_f) {
            if (matrix_a.m_rows != vec_f.m_rows) {
                throw LinAlgError(const_cast<char*>("bad shape."));
            }
            T main_det = matrix_a.det();
            if (main_det == 0) {
                throw LinAlgError(const_cast<char*>("det = 0."));
            }
            Matrix<double> result(matrix_a.m_rows, 1);
            for (int i = 0; i < result.m_rows; i++) {
                Matrix<T> minor_mat(matrix_a);
                for (int j = 0; j < result.m_rows; j++) {
                    minor_mat(j, i) = vec_f(j, 0);
                }
                T det_minor = minor_mat.det();
                result(i, 0) = det_minor / main_det;
            }
            return result;
	    }
        template<typename T_new>
        friend struct TextMode;
        template<typename T_new>
        friend struct BinaryMode;
    };
    template<typename T_new>
    struct TextMode {
        static void write(const char* file_name, const linalg::Matrix<T_new>& object) { 
            std::ofstream in(file_name); 
            in << "type: " << typeid(T_new).name();
            in << "\nrows: " << object.m_rows << "\ncolumns : " << object.m_columns << "\n" << object;
            in.close(); 
        }
    };
    template<typename T_new>
    struct BinaryMode {
        static void write(const char* file_name, const linalg::Matrix<T_new>& object) { 
            std::ofstream in(file_name); 
            in << "type: " << typeid(T_new).name();
            in << "\nrows: " << object.m_rows << "\ncolumns : " << object.m_columns << "\n" << object;
            in.close(); 
        }
    };

} ///namespace linalg.

template<typename T = double>
class Complex {
    T re, im;
public:
    Complex() : re(0), im(0) { }
    Complex(T r) : re(r), im(0) { } 
    Complex(T r, T i) : re(r), im(i) { }

    T real() const noexcept {
        return re;
    }
    T imag() const noexcept {
        return im;
    }
    void real(T real) {
        re = real;
    }
    void imag(T imag) {
        im = imag;
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T>& operator=(T_new num) {
        re = num;
        im = 0;
        return *this; 
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T>& operator=(const Complex<T_new> &other) {
        re = other.real();
        im = other.imag();
        return *this; 
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    void operator+=(T_new num) {
        re += num;
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    void operator-=(T_new num) {
        re -= num;
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    void operator*=(T_new num) {
        re *= num;
        im *= num;
    }
    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    void operator+=(Complex<T_new> num) {
        re += num.real();
        im += num.imag();
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    void operator-=(Complex<T_new> num) {
        re -= num.real();
        im -= num.imag();
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    void operator*=(Complex<T_new> &other) {
        auto t_re = re;
        re = re * other.re - im * other.im;
        im = t_re * other.re + im * other.im;
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator+(T_new num) {
        return Complex<typename std::common_type<T, T_new>::type>{re + num, im};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator-(T_new num) {
        return Complex<typename std::common_type<T, T_new>::type>{re - num, im};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator*(T_new num) {
        return Complex<T>{static_cast<T>(re * num), static_cast<T>(im * num)};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator*(const Complex<T_new> &second) {
        return Complex<T>{static_cast<T>(re * second.real() - im * second.imag()),
                             static_cast<T>(re * second.imag() + im * second.real())};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    friend Complex<T> operator*(T_new num, const Complex<T> &second) {
        return Complex<T>{static_cast<T>(second.real() * num), static_cast<T>(second.imag() * num)};
    }


    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    friend Complex<T> operator+(T_new num, Complex<T> &other) {
        return Complex<typename std::common_type<T, T_new>::type>{other.real() + num, other.imag()};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    friend Complex<T> operator-(T_new num, Complex<T> &other) {
        return Complex<typename std::common_type<T, T_new>::type>{other.real() - num, other.imag()};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    friend Complex<T> operator*(T_new num, Complex<T> &other) {
        return Complex<typename std::common_type<T, T_new>::type>{other.real() * num, other.imag() * num};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator*(const Complex<T> &other) {
        return Complex<T>{static_cast<T>(re * other.real() - im * other.imag()),
                             static_cast<T>(re * other.imag() + im * other.real())};
    }

    // template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    // friend Complex<T> operator*(T_new num, const Complex<T> &other) {
    //     return Complex<T>{static_cast<T>(other.real() * num), static_cast<T>(other.imag() * num)};
    // }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator+(const Complex<T_new> &other) {
        return Complex<typename std::common_type<T, T_new>::type>{re + other.real(), im + other.imag()};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator-(const Complex<T_new> &other) {
        return Complex<typename std::common_type<T, T_new>::type>{re - other.real(), im - other.imag()};
    }

    // template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    // Complex<T> operator*(const Complex<T_new> &other) {
    //     return Complex<typename std::common_type<T, T_new>::type>{re * other.real() - im * other.imag(), re * other.real() + im * other.imag()};
    // }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator/(T_new num) {
        return Complex<T>{static_cast<T>(re / num), static_cast<T>(im / num)};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    Complex<T> operator/(const Complex<T_new> &second) {
        return Complex<T>{static_cast<T>((re * second.real() + im * second.imag()) /
                                               (second.real() * second.real() + second.imag() * second.imag())),
                             static_cast<T>((-re * second.imag() + im * second.real()) /
                                               (second.real() * second.real() + second.imag() * second.imag()))};
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    friend Complex<T> operator/(T_new num, const Complex<T> &second) {
        return Complex<T>{static_cast<T>(num / second.real()), static_cast<T>(num / second.imag())};
    }


    friend std::ostream &operator<<(std::ostream &out, const Complex<T> &other) {
        if (other.real() != 0) {
            out << other.real();
        }
        if (other.imag() > 0) {
            out << " + " << other.imag() << "i";
        }
        if (other.imag() < 0) {
            out << " - " << other.imag() * (-1) << "i";
        }
        return out;
    }

    friend std::istream &operator>>(std::istream &in, Complex<T> &other) {
        std::string l;
        getline(in, l);
        auto p1 = std::min(l.substr(1).rfind("+"), l.substr(1).rfind("-")) + 1;
        auto p2 = l.rfind("*");
        other.real(atof(l.substr(0, p1).c_str()));
        other.imag(atof(l.substr(p1, p2 - p1).c_str()));
        return in;
    }

    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    bool operator==(T_new num) {
        return (re == static_cast<T>(num)  && im == 0);
    }

    template<typename T_new>
    bool operator==(const Complex<T_new> &second) {
        return (re == static_cast<T>(second.real()) && im == static_cast<T>(second.imag()));
    }
    template<typename T_new, typename = typename std::enable_if<std::is_arithmetic<T_new>::value, T_new>::type>
    bool operator!=(T_new num) {
        return !(*this == num);
    }

    template<typename T_new>
    bool operator!=(const Complex<T_new> &second) {
        return (re != static_cast<T>(second.real()) || im != static_cast<T>(second.imag()));
    }

};
