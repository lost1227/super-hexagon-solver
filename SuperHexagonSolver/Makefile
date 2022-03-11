CXX = g++
CPPFLAGS = -Wall -Werror -pedantic -std=c++2a -fPIC

CPPFLAGS += -Wno-error=sign-compare

CPPFLAGS += -g -O0
# CPPFLAGS += -O3

CPPFLAGS += -DLITTLE_ENDIAN
CPPFLAGS += -fsanitize=undefined -fsanitize=address

SRCS = $(shell find . -name "*.cpp")
OBJS = $(SRCS:.cpp=.o)
MAIN = hexagon_solver

DEPDIR := .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

CV_DIR = /home/jordan/Documents/git/opencv/opencv

INCLUDES = -I$(CV_DIR)/build \
	-I$(CV_DIR)/modules/core/include \
	-I$(CV_DIR)/modules/imgcodecs/include \
	-I$(CV_DIR)/modules/highgui/include \
	-I$(CV_DIR)/modules/videoio/include \
	-I$(CV_DIR)/modules/imgproc/include

LIBDIRS = -L$(CV_DIR)/build/lib
LIBRARIES = -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lopencv_imgproc \
	-lX11

LINKFLAGS = -static-libstdc++ -Wl,-rpath $(CV_DIR)/build/lib

COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(TARGET_ARCH) -c

%.o : %.c $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

all: $(MAIN)

$(MAIN) : $(OBJS)
	$(CXX) $(LINKFLAGS) $(CPPFLAGS) $(LIBDIRS) $(LIBRARIES) $(OBJS) -o $(MAIN)

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):
include $(wildcard $(DEPFILES))

clean:
	rm -f $(OBJS) $(MAIN) $(DEPFILES)
