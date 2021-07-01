#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <complex>
#include <valarray>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>


/**
 * Returns if the given args conatinig the given arg
 */
bool has_arg(
    int argc, 
    char *argv[], 
    const char *short_arg, 
    const char *long_arg
);

/**
 * Returns the given argument arg
 */
std::string get_arg(
    int argc, 
    char *argv[], 
    const char *short_arg, 
    const char *long_arg
);

/**
 * Returns the given argument index
 */
int get_arg_i(
    int argc, 
    char *argv[], 
    const char *short_arg, 
    const char *long_arg
);

/**
 * shorter macros to use in main
 */
#define has_arg1(long_arg) has_arg(argc, argv, NULL, long_arg)
#define has_arg2(short_arg, long_arg) has_arg(argc, argv, short_arg, long_arg)
#define has_arg4(argc, argv, short_arg, long_arg) \
  has_arg(argc, argv, short_arg, long_arg)
#define has_argx(X, T1, T2, T3, T4, FUNC, ...) FUNC
#define has_arg(...) has_argx(, ##__VA_ARGS__,            \
                                has_arg4(__VA_ARGS__),    \
                                "3 args not allowed",     \
                                has_arg2(__VA_ARGS__),    \
                                has_arg1(__VA_ARGS__))

#define get_arg1(long_arg) get_arg(argc, argv, NULL, long_arg)
#define get_arg2(short_arg, long_arg) get_arg(argc, argv, short_arg, long_arg)
#define get_arg4(argc, argv, short_arg, long_arg) \
  get_arg(argc, argv, short_arg, long_arg)
#define get_argx(X, T1, T2, T3, T4, FUNC, ...) FUNC
#define get_arg(...) get_argx(, ##__VA_ARGS__,            \
                                get_arg4(__VA_ARGS__),    \
                                "3 args not allowed",     \
                                get_arg2(__VA_ARGS__),    \
                                get_arg1(__VA_ARGS__))

#define get_arg_i1(long_arg) get_arg_i(argc, argv, NULL, long_arg)
#define get_arg_i2(short_arg, long_arg) get_arg_i(argc, argv, short_arg, long_arg)
#define get_arg_i4(argc, argv, short_arg, long_arg) \
  get_arg_i(argc, argv, short_arg, long_arg)
#define get_arg_ix(X, T1, T2, T3, T4, FUNC, ...) FUNC
#define get_arg_i(...) get_arg_ix(, ##__VA_ARGS__,              \
                                  get_arg_i4(__VA_ARGS__),      \
                                  "3 args not allowed",         \
                                  get_arg_i2(__VA_ARGS__),      \
                                  get_arg_i1(__VA_ARGS__))

/**
 * to get integer args
 */
#define get_i_arg(...)                                                    \
  (has_arg(__VA_ARGS__) ? atoi(get_arg(__VA_ARGS__).c_str()) : -1)

#define get_l_arg(...)                                                    \
  (has_arg(__VA_ARGS__) ? atol(get_arg(__VA_ARGS__).c_str() : -1)

#define get_ll_arg(...)                                                   \
  (has_arg(__VA_ARGS__) ? atoll(get_arg(__VA_ARGS__).c_str() : -1)

/**
 * to get float args
 */
#define get_f_arg(...)                                                    \
  (has_arg(__VA_ARGS__) ? atof(get_arg(__VA_ARGS__).c_str()) : -1.0)

/* returns the given argument or the default argument */
#define parse_arg(DEFAULT, ...) (has_arg(__VA_ARGS__) ? get_arg(__VA_ARGS__) : (DEFAULT))
#define parse_i_arg(DEFAULT, ...) (has_arg(__VA_ARGS__) ? atoll(get_arg(__VA_ARGS__).c_str()) : (DEFAULT))
#define parse_f_arg(DEFAULT, ...) (has_arg(__VA_ARGS__) ? atof(get_arg(__VA_ARGS__).c_str()) : (DEFAULT))

/* returns the current milliseconds time */
uint64_t gettime_usec();

inline uint32_t rand32(uint32_t *x) {
    *x ^= *x << 13;
    *x ^= *x >> 17;
    *x ^= *x << 5;
    return *x;
}

/* random value */
typedef struct {
    uint32_t x, y, z, w;
} rand128_t; 

inline uint32_t rand128(rand128_t *rand) {
                                 
    uint32_t t = rand->x ^ (rand->x << 11);
    rand->x = rand->y; rand->y = rand->z; rand->z = rand->w;
    rand->w ^= (rand->w >> 19) ^ t ^ (t >> 8);
                                        
    return rand->w;
}

uint32_t rand128();

/* initializes the global rand128 with the given seed */
void init_rand128(uint32_t seed);

/* creates a new random variable based of a seed from the global rand */
rand128_t *new_rand128_t();
rand128_t *new_rand128_t(uint32_t seed);
                      

inline uint32_t rand_range128(rand128_t *rand, uint32_t start, uint32_t end) {
    return (start + (rand128(rand) % (((end) + 1) - (start))));
}
inline uint32_t rand_range128(uint32_t start, uint32_t end) {
    return (start + (rand128() % (((end) + 1) - (start))));
}

inline FloatType rand_range_d128(rand128_t *rand, FloatType start, FloatType end) {
    return ((FloatType(rand128(rand)) / 4294967295.0) * (FloatType(end) - FloatType(start)) + FloatType(start));
}
inline FloatType rand_range_d128(FloatType start, FloatType end) {
    return ((FloatType(rand128()) / 4294967295.0) * (FloatType(end) - FloatType(start)) + FloatType(start));
}

/* returns normal distributed random value */
inline double rand128n(rand128_t *rand) {
    static unsigned counter = 1;
    static double z0 = 0.0, z1 = 0.0, u = 0.0, v = 0.0, s = 0.0;

    counter++;
    if (counter % 2 == 0) {
        do {
            u = rand_range_d128(rand, -1.0, 1.0);
            v = rand_range_d128(rand, -1.0, 1.0);
            s = pow(u, 2) + pow(v, 2);
        } while (s >= 1.0 || s == 0.0);

        z0 = u * sqrt((-2 * log(s)) / s);
        z1 = v * sqrt((-2 * log(s)) / s);
        return z0;
    }

    return z1;    
}
inline double rand128n() {
    static unsigned counter = 1;
    static double z0 = 0.0, z1 = 0.0, u = 0.0, v = 0.0, s = 0.0;

    counter++;
    if (counter % 2 == 0) {
        do {
            u = rand_range_d128(-1.0, 1.0);
            v = rand_range_d128(-1.0, 1.0);
            s = pow(u, 2) + pow(v, 2);
        } while (s >= 1.0 || s == 0.0);

        z0 = u * sqrt((-2 * log(s)) / s);
        z1 = v * sqrt((-2 * log(s)) / s);
        return z0;
    }

    return z1;    
}

inline double rand128n(rand128_t *rand, double mean, double stddev) {

    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached) {
        double x, y, r;
        do {
            x = rand_range_d128(rand, -1.0, 1.0);
            y = rand_range_d128(rand, -1.0, 1.0);

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}
inline double rand128n(double mean, double stddev) {

    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached) {
        double x, y, r;
        do {
            x = rand_range_d128(-1.0, 1.0);
            y = rand_range_d128(-1.0, 1.0);

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

/* creats a exponential distributed random value with the given rate */
#define rand128exp(rand, rate) (-1.0 * log(rand_range_d128(rand, 0.0, 1.0)) / double(rate))
#define rand128exp_u(rand, rate) unsigned(round(rand128exp(rand, rate)))

/**
 * Splits a String by a specific char
 * returns vector<string> splitted Strings
 */
std::vector<std::string> split(const std::string str, const std::string seperator);

/**
 * converts a given integer to a string
 */
std::string itoa(int64_t i);

/**
 * converts a given double to a string
 */
std::string ftoa(double d, unsigned precision = 3);

/**
 * converts a boolean to a string
 */
std::string btoa(bool b);


/* generates an unique id */
uint64_t genUID();

/* macro for filename without path */
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/* set the overall log level */
void set_log_level(unsigned log_level);

/* enable / disable fail on warnings */
void fail_on_warning(bool fail);

/* set the overall log file */
void set_log_file(int logfd);

/* log levels */
#define LOG_DD 5 /* debug level 2      */
#define LOG_D  4 /* debug level 1      */
#define LOG_I  3 /* information output */
#define LOG_W  2 /* warning            */
#define LOG_E  1 /* error              */
#define LOG_EE 0 /* fatal error        */
#define NO_LOG 9 /* no loging          */

/* returns the last error / warn message */
std::string get_last_error();

/* prints a syncronized message */
void log_str(std::string msg, unsigned log_level = LOG_I);

#define log_err(str, status) \
  log_str(std::string("[") + __FILENAME__ + ":" + itoa(__LINE__) + "] " + str, status)

/* writes the given value to the given file */
#define writeValue(fd, value, size)                         \
    if (write(fd, value, size) != ssize_t(size)) {          \
        log_err("failed to write value to file", LOG_E);    \
    }

/* reads the given value from the given file */
#define readValue(fd, value, size)                           \
    if (read(fd, value, size) != ssize_t(size)) {            \
        log_err("failed to read value from file", LOG_E);    \
    }

/* sign function */
template <typename T> T sgn(T x) { return (x < 0) ? T(-1) : T(1); }

/* returns true wether the given string ends with the given substring */
bool endsWith(std::string str, std::string substr);

/* creats a plot from the given x and y vector */
void saveLaTeXPlot(
    std::vector<FloatType> xValues, 
    std::vector<FloatType> yValues, 
    std::string filename,
    std::string xlabel,
    std::string ylabel
);

/* creats a plot from the given x and y vector */
void saveLaTeXRangePlot(
    std::vector<FloatType> xValues, 
    std::vector<FloatType> yValuesMean, 
    std::vector<FloatType> yValuesStd, 
    std::string filename,
    std::string xlabel,
    std::string ylabel
);

/* saves the given 2D vector as image */
void saveAsImage(
    std::string name, 
    std::vector<std::vector<FloatType>> data,
    unsigned magnification = 1
);

/* variable arg min function */
template <typename T1, typename T2> T1 vmin(T1 v1, T2 v2) {
    return (v1 <= v2) ? v1 : v2;
}
template <typename T1, typename T2, typename ...TN> T1 vmin(T1 v1, T2 v2, TN...vn) {
    return vmin(v1, vmin(v2, vn...));
}

/* variable arg max function */
template <typename T1, typename T2> T1 vmax(T1 v1, T2 v2) {
    return (v1 >= v2) ? v1 : v2;
}
template <typename T1, typename T2, typename ...TN> T1 vmax(T1 v1, T2 v2, TN...vn) {
    return vmax(v1, vmax(v2, vn...));
}

/* running average and standart derivation */
class RunningStatistics {
    
    private:

        /* the memory of this */
        FloatType memory;

        /* runing mean */
        FloatType mean;

        /* runing varianz */
        FloatType var;

        /* running sum of smaples */
        FloatType sum;

    public:

        /* constructor */
        RunningStatistics(FloatType memory = 10) : memory(memory), mean(0), var(0), sum(0) { }

        /* adds a sample to this */
        inline void add(FloatType sample, FloatType memory = 0) {
            
            FloatType decay = 1.0 - ((memory > 0) ? 1.0 / memory : 1.0 / this->memory);

            mean = decay * mean + sample;
            sum  = decay * sum  + 1;
            var  = decay * var  + pow(getMean() - sample, 2);
        }

        /* returns the mean of this */
        inline FloatType getMean() { return mean / sum; }

        /* returns the varianz of this */
        inline FloatType getVar() { return var / sum; }

        /* returns the standart derivation o this */
        inline FloatType getSTD() { return sqrt(getVar()); }

        /* returns the number samples currently considered */
        inline FloatType getSum() { return sum; }

};

/* registers a SignalHandler */
class SignalHandler;
void registerSignalHandler(SignalHandler *handler);

/* interface for signal handling */
class SignalHandler {

    public: 
    
        /* can execute an action on keyboard interrupt (STRG+C) */
        virtual void interupt() = 0;

        /* virtual destructor for correct polymorphism */
        virtual ~SignalHandler() { }

        /* register this as a signal handler */
        SignalHandler() { registerSignalHandler(this); }
};

namespace std {
    double min(float x, double y);
    double min(double x, float y);
    double min(int x, float y);
    double min(float x, int y);
    double min(int x, double y);
    double min(double x, int y);
}

/* converts the given degree into radian */
#define toRadian(deg) ((deg) * 0.017453292519943295)

/* convert orientation vectors to and from quaternions */
void toQuaternion(
    FloatType xUp, 
    FloatType yUp, 
    FloatType zUp,
    FloatType xXDirection, 
    FloatType yXDirection, 
    FloatType zXDirection,
    FloatType &qw,
    FloatType &qx,
    FloatType &qy,
    FloatType &qz
);
void toVectors(
    FloatType &xUp, 
    FloatType &yUp, 
    FloatType &zUp,
    FloatType &xXDirection, 
    FloatType &yXDirection, 
    FloatType &zXDirection,
    FloatType qw,
    FloatType qx,
    FloatType qy,
    FloatType qz
);

#endif /* __UTILS_H__ */
