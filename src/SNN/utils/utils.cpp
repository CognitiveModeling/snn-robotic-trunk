/**
 * Some utility functions
 */
#include "utils.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <inttypes.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <math.h>
#include "Image.h"
#include <csignal>

using std::vector;
using std::string;
using SNN::Image;

static rand128_t *globalRand = NULL;

rand128_t *new_rand128_t(uint32_t seed) {
    rand128_t *rand = (rand128_t *) malloc(sizeof(rand128_t));
 
    rand->x = rand32(&seed);
    rand->y = rand32(&seed);
    rand->z = rand32(&seed);
    rand->w = rand32(&seed);
 
    return rand;
}

/* initializes the global rand128 with the given seed */
void init_rand128(uint32_t seed) {
    if (globalRand != NULL) {
        fprintf(stderr, "Error rand already initialized");
        assert(false);
        exit(EXIT_FAILURE);
    }
    globalRand = new_rand128_t(seed);
}

/* creates a new random variable based of a seed from the global rand */
rand128_t *new_rand128_t() {
    if (globalRand == NULL) {
        uint32_t seed = gettime_usec();
        printf("random seed: %u\n", seed);
        globalRand = new_rand128_t(seed);
    }

    rand128_t *rand = (rand128_t *) malloc(sizeof(rand128_t));
 
    rand->x = rand128(globalRand);
    rand->y = rand128(globalRand);
    rand->z = rand128(globalRand);
    rand->w = rand128(globalRand);
 
    return rand;
}


uint32_t rand128() { 
    if (globalRand == NULL) {
        uint32_t seed = gettime_usec();
        printf("random seed: %u\n", seed);
        globalRand = new_rand128_t(seed);
    }
    return rand128(globalRand); 
}

/**
 * Returns if the given args conatinig the given arg
 */
bool has_arg(
    int argc, 
    char *argv[], 
    const char *short_arg, 
    const char *long_arg
) {

    int i;
    for (i = 1; i < argc; i++) {
        if ((short_arg != NULL && !strcmp(argv[i], short_arg)) ||
            (long_arg != NULL  && !strcmp(argv[i], long_arg))) {
          
            return true;
        } else if (strlen(argv[i]) == 2 && !strcmp(argv[i], "--"))
            return false;
    }
 
    return false;
}

/**
 * Returns the given argument arg
 */
string get_arg(
    int argc, 
    char *argv[], 
    const char *short_arg, 
    const char *long_arg
) {

    int i;
    for (i = 1; i < argc - 1; i++) {
        if ((short_arg != NULL && !strcmp(argv[i], short_arg)) ||
            (long_arg != NULL  && !strcmp(argv[i], long_arg))) {
          
            return string(argv[i + 1]);
        } else if (strlen(argv[i]) == 2 && !strcmp(argv[i], "--"))
            return string("");
    }
 
    return string("");
}

/**
 * Returns the given argument index
 */
int get_arg_i(
    int argc, 
    char *argv[], 
    const char *short_arg, 
    const char *long_arg
) {

    int i;
    for (i = 1; i < argc - 1; i++) {
        if ((short_arg != NULL && !strcmp(argv[i], short_arg)) ||
            (long_arg != NULL  && !strcmp(argv[i], long_arg))) {
          
            return i + 1;
        } else if (strlen(argv[i]) == 2 && !strcmp(argv[i], "--"))
            return -1;
    }
 
    return -1;
}

/* returns the current milliseconds time */
uint64_t gettime_usec() {

    struct timeval time;
    if (gettimeofday(&time, NULL) == -1) {
        return ((uint64_t) -1);
    }

    return ((uint64_t) time.tv_sec) * ((uint64_t) 1000000) + 
           ((uint64_t) time.tv_usec);
}

/**
 * Splits a String by a specific char
 * returns vector<string> splitted Strings
 */
vector<string> split(const string str, const string seperator) {

  char *copy = strdup(str.c_str());
  char *ptr = strtok(copy, seperator.c_str());

  vector<string> res;

  int i;
  for (i = 0; ptr != NULL; i++) {

    int size = strlen(ptr);
    res.push_back(string(ptr, size));
    ptr = strtok(NULL, seperator.c_str());
  }

  free(copy);
  return res;
}

/**
 * converts a given integer to a string
 */
string itoa(int64_t i) {

  char a[32];

  if (i == 0) {
    a[0] = '0';
    a[1] = '\0';
    return string(a);
  }

  char is_neg = (i < 0);

  if (is_neg)
    i *= -1;

  int64_t j;
  for (j = 0; i != 0; j++, i /= 10)
    a[j] = 48 + (i % 10);

  if (is_neg) {
     a[j] = '-';
    j++;
  }

  a[j] = '\0';
  string str = string(a);;
  return string(str.rbegin(), str.rend());
}

/**
 * converts a given double to a string
 */
string ftoa(double d, unsigned precision) {
    
  /* check for NaN */
  if (isnan(d))
      return string("NaN");

  /* check for infinity */
  if (isinf(d) && d < 0.0)
      return string("-inf");
  if (isinf(d) && d > 0.0)
      return string("inf");


  string res = (d < 0.0) ? "-" : "";

  if (d < 0.0)
    d *= -1.0;

  if (d > double(UINT64_MAX)) {
    unsigned exp = 0;
    while (d > 10) {
        d *= 0.1;
        exp++;
    }
    return ftoa(d, precision) + "e+" + itoa(exp);
  }

  res += itoa(d) + ".";


  double decimal = (d - ((double) ((uint64_t) d))) * 10;

  for (unsigned i = 0; i < precision; i++) {
    res += itoa(decimal);
    decimal = decimal - ((double) ((uint64_t) decimal));
    decimal *= 10;
  }
    
  return res;
}

/**
 * converts a boolean to a string
 */
string btoa(bool b) {
  if (b) return string("true");
  return string("false");
}


/* generates an unique id */
uint64_t genUID() {
    static uint64_t id = 0;
    return id++;
}

/* the overall log level */
static unsigned loglevel = LOG_I;

/* wether to fail on warning s */
static bool failOnWarning = false;

/* the overall log file */
static int log_file = STDOUT_FILENO;

/* set the overall log level */
void set_log_level(unsigned log_level) {
    loglevel = log_level;
}

/* enable / disable fail on warnings */
void fail_on_warning(bool fail) {
    failOnWarning = fail;
}

/* set the overall log file */
void set_log_file(int logfd) {
    log_file = logfd;
}

/* the last error / warn message */
static string last_err_msg = "";

/* returns the last error / warn message */
string get_last_error() {
    return last_err_msg;
}

#define write_str(fd, str) write(fd, str.c_str(), str.length())


/* prints a syncronized message */
void log_str(string msg, unsigned log_level) {

  if (loglevel == NO_LOG)
      return;

  if (loglevel >= log_level) {
    
    int ret = -1;
    switch (log_level) {
      case LOG_DD : ret = write_str(log_file, string("\r[D2]")); break;
      case LOG_D  : ret = write_str(log_file, string("\r[D1]")); break;
      case LOG_I  : ret = write_str(log_file, string("\r[II]")); break;
      case LOG_W  : ret = write_str(log_file, string("\r[WW]")); break;
      case LOG_E  : ret = write_str(log_file, string("\r[EE]")); break;
      case LOG_EE : ret = write_str(log_file, string("\r[FE]")); break;
      default     : ret = write_str(log_file, string("\r[UU]")); break;
    }

    //ret = write_str(log_file, string(" (" + itoa(gettime_usec()) + ") "));
    ret = write_str(log_file, string(" "));
    ret = write_str(log_file, msg);
    ret = write_str(log_file, string("\n"));

    if (log_level < LOG_I)
        last_err_msg = msg;

    if (failOnWarning && log_level == LOG_W)
        assert(false);

    if (log_level == LOG_E)
      assert(false);
 
    if (log_level == LOG_EE)
      abort();

    assert(ret > 0);
  } 
}

/* returns true wether the given string ends with the given substring */
bool endsWith(std::string str, std::string substr) {
    if (str.length() < substr.length()) 
        return false;

    const int offset = str.length() - substr.length();
    for (unsigned i = offset; i < str.length(); i++)
        if (str[i] != substr[i - offset])
            return false;

    return true;
}

/* creats a plot from the given x and y vector */
void saveLaTeXPlot(
    std::vector<FloatType> xValues, 
    std::vector<FloatType> yValues, 
    std::string filename,
    std::string xlabel,
    std::string ylabel
) {

    FILE *file = fopen(filename.c_str(), "w");
    if (file != NULL) {
        
        fprintf(file, 
            "\\documentclass[10pt]{beamer}\n"
            "\\usetheme{metropolis}\n"
            "\\usepackage{appendixnumberbeamer}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage[scale=2]{ccicons}\n"
            "\\usepackage[ngerman]{babel}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{pgfplots}\n"
            "\\usepgfplotslibrary{dateplot}\n"
            "\\usepgfplotslibrary{fillbetween}\n"
            "\\usepackage{xspace}\n"
            "\\newcommand{\\themename}{\\textbf{\\textsc{metropolis}}\\xspace}\n"
            "\\usepackage[loop,autoplay]{animate}\n"
            "\\begin{document}\n\n"
            "\\begin{figure}[ht]\n"
            "  \\begin{tikzpicture}\n"
            "    \\begin{axis}[\n"
            "      style={font=\\scriptsize},\n"
            "      width=\\textwidth,\n"
            "      height=7cm,\n"
            "      grid=both,\n"
            "      minor y tick num=4,\n"
            "      minor x tick num=1,\n"
            "      major tick length=0pt,\n"
            "      minor tick length=0pt,\n"
            "      legend pos=south east,\n"
            "      xlabel={%s},\n"
            "      ylabel={%s}\n"
            "    ]\n"
            "        \\addplot+[mark=none,black] plot coordinates {\n",
            xlabel.c_str(), ylabel.c_str()
        );

        for (unsigned i = 0; i < std::min(xValues.size(), yValues.size()); i++)
            fprintf(file, "(%f,%f)\n", xValues[i], yValues[i]);

        fprintf(file, " };\n"
            "    \\end{axis}\n"
            "  \\end{tikzpicture}\n"
            "\\end{figure}\n"
            "\\end{document}\n"
        );
        fclose(file);
    }
}

/* creats a plot from the given x and y vector */
void saveLaTeXRangePlot(
    std::vector<FloatType> xValues, 
    std::vector<FloatType> yValuesMean, 
    std::vector<FloatType> yValuesStd, 
    std::string filename,
    std::string xlabel,
    std::string ylabel
) {

    FILE *file = fopen(filename.c_str(), "w");
    if (file != NULL) {
        
        fprintf(file, 
            "\\documentclass[10pt]{beamer}\n"
            "\\usetheme{metropolis}\n"
            "\\usepackage{appendixnumberbeamer}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage[scale=2]{ccicons}\n"
            "\\usepackage[ngerman]{babel}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{pgfplots}\n"
            "\\usepgfplotslibrary{dateplot}\n"
            "\\usepgfplotslibrary{fillbetween}\n"
            "\\usepackage{xspace}\n"
            "\\newcommand{\\themename}{\\textbf{\\textsc{metropolis}}\\xspace}\n"
            "\\usepackage[loop,autoplay]{animate}\n"
            "\\begin{document}\n\n"
            "\\begin{figure}[ht]\n"
            "  \\begin{tikzpicture}\n"
            "    \\begin{axis}[\n"
            "      style={font=\\scriptsize},\n"
            "      width=\\textwidth,\n"
            "      height=7cm,\n"
            "      grid=both,\n"
            "      minor y tick num=4,\n"
            "      minor x tick num=1,\n"
            "      major tick length=0pt,\n"
            "      minor tick length=0pt,\n"
            "      legend pos=south east,\n"
            "      xlabel={%s},\n"
            "      ylabel={%s}\n"
            "    ]\n"
            "        \\addplot+[mark=none,name path=Upper,blue!50,forget plot] plot coordinates {",
            xlabel.c_str(), ylabel.c_str()
        );

        unsigned length = std::min(xValues.size(), std::min(yValuesMean.size(), yValuesStd.size()));

        for (unsigned i = 0; i < length; i++)
            fprintf(file, " (%f,%f)", xValues[i], yValuesMean[i] + yValuesStd[i]);

        fprintf(file, " };\n"
            "        \\addplot+[mark=none,black,forget plot] plot coordinates {");

        for (unsigned i = 0; i < length; i++)
            fprintf(file, " (%f,%f)", xValues[i], yValuesMean[i]);

        fprintf(file, " };\n"
            "        \\addplot+[mark=none,name path=Lower,blue!50,forget plot] plot coordinates {");

        for (unsigned i = 0; i < length; i++)
            fprintf(file, " (%f,%f)", xValues[i], yValuesMean[i] - yValuesStd[i]);

        fprintf(file, " };\n"
            "        \\addplot[blue!50] fill between[of=Upper and Lower];\n"
            "    \\end{axis}\n"
            "  \\end{tikzpicture}\n"
            "\\end{figure}\n"
            "\\end{document}\n"
        );
        fclose(file);
    }
}

/* saves the given 2D vector as image */
void saveAsImage(
    std::string name, 
    std::vector<std::vector<FloatType>> data,
    unsigned magnification 
) {
    Image img(data.size(), data[0].size(), 3);
    
    FloatType meanPos = 0, meanNeg = 0;
    FloatType meanPosSqr = 0, meanNegSqr = 0;
    unsigned meanNegValues = 0, meanPosValues = 0;
    for (unsigned x = 0; x < data.size(); x++) {
        for (unsigned y = 0; y < data[0].size(); y++) {
            if (data[x][y] > 0) {
                meanPos    += data[x][y];
                meanPosSqr += pow(data[x][y], 2);
                meanPosValues++;
            }
            if (data[x][y] < 0) {
                meanNeg    += data[x][y];
                meanNegSqr += pow(data[x][y], 2);
                meanNegValues++;
            }
        }
    }
    FloatType stdPos = 0, stdNeg = 0;
    if (meanPosValues > 1) {
        meanPos    /= meanPosValues;
        meanPosSqr /= meanPosValues;
        stdPos = sqrt((FloatType(meanPosValues) / (meanPosValues - 1)) * (meanPosSqr - pow(meanPos, 2)));
    }
    if (meanNegValues > 1) {
        meanNeg    /= meanNegValues;
        meanNegSqr /= meanNegValues;
        stdNeg = sqrt((FloatType(meanNegValues) / (meanNegValues - 1)) * (meanNegSqr - pow(meanNeg, 2)));
    }
    printf("num +: %u, mean +: %f, std +: %f, num -: %u mean -: %f, std -: %f\n", meanPosValues, meanPos, stdPos, meanNegValues, meanNeg, stdNeg);
    
    for (unsigned x = 0; x < data.size(); x++) {
        for (unsigned y = 0; y < data[0].size(); y++) {
            if (data[x][y] > 0)
                img.getPixelGreen(x, y) = std::min(unsigned(255 * data[x][y] / (meanPos + 2 * stdPos)), 255u);
            if (data[x][y] < 0)
                img.getPixelRed(x, y) = std::min(unsigned(255 * data[x][y] / (meanNeg - 2 * stdNeg)), 255u);
        }
    }

    if (magnification > 1)
        img.upscal(magnification);
    img.save(name);
}

/* vector of signal handlers */
static vector<SignalHandler *> signalHandlers;

/* function to handle keyboard interupt signals */
void signalHandlerFunction(int signum) {

    static uint64_t lastCallTime = 0;
    
    /* exit if pressed twice within one second */
    if (gettime_usec() - lastCallTime < 1000000) {
        log_str("Keyboard Interrupt exiting", LOG_I);
        exit(signum);  
    }

    log_str("Keyboard Interrupt saving values, press fast to exit", LOG_I);
    lastCallTime = gettime_usec();

    for (auto &handler: signalHandlers)
        handler->interupt();
}

/* registers a SignalHandler */
void registerSignalHandler(SignalHandler *handler) {
	if (signalHandlers.empty())
		signal(SIGINT, signalHandlerFunction);  

    signalHandlers.push_back(handler);
}

namespace std {
    double min(float x, double y) { return min<double>(x, y); }
    double min(double x, float y) { return min<double>(x, y); }
    double min(int x, float y) { return min<double>(x, y); }
    double min(float x, int y) { return min<double>(x, y); }
    double min(int x, double y) { return min<double>(x, y); }
    double min(double x, int y) { return min<double>(x, y); }
}

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
) {
    double a[3][3] = {
        { xXDirection, xUp, yXDirection * zUp - zXDirection * yUp }, 
        { yXDirection, yUp, zXDirection * xUp - xXDirection * zUp }, 
        { zXDirection, zUp, xXDirection * yUp - yXDirection * xUp } 
    };

	double trace = a[0][0] + a[1][1] + a[2][2]; 
	if(trace > 0) {
	    double s = 0.5 / sqrt(trace + 1.0);
	    qw = 0.25 / s;
	    qx = (a[2][1] - a[1][2]) * s;
	    qy = (a[0][2] - a[2][0]) * s;
	    qz = (a[1][0] - a[0][1]) * s;
	} else {
	    if (a[0][0] > a[1][1] && a[0][0] > a[2][2]) {
	        double s = 2.0 * sqrt(1.0 + a[0][0] - a[1][1] - a[2][2]);
	        qw = (a[2][1] - a[1][2]) / s;
	        qx = 0.25 * s;
	        qy = (a[0][1] + a[1][0]) / s;
	        qz = (a[0][2] + a[2][0]) / s;
	    } else if (a[1][1] > a[2][2]) {
	        double s = 2.0 * sqrt(1.0 + a[1][1] - a[0][0] - a[2][2]);
	        qw = (a[0][2] - a[2][0]) / s;
	        qx = (a[0][1] + a[1][0]) / s;
	        qy = 0.25 * s;
	        qz = (a[1][2] + a[2][1]) / s;
	    } else {
	        double s = 2.0 * sqrt(1.0 + a[2][2] - a[0][0] - a[1][1]);
	        qw = (a[1][0] - a[0][1]) / s;
	        qx = (a[0][2] + a[2][0]) / s;
	        qy = (a[1][2] + a[2][1]) / s;
	        qz = 0.25 * s;
	    }
	}
}
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
) {
    xXDirection = 1 - 2* qy*qy - 2* qz*qz;
    yXDirection = 2*qx*qy + 2*qz*qw;
    zXDirection = 2*qx*qz - 2*qy*qw;

    xUp = 2*qx*qy - 2*qz*qw;
    yUp = 1 - 2*qx*qx - 2*qz*qz;
    zUp = 2*qy*qz + 2*qx*qw;
}
