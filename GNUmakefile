HOST_OS		:= $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS	?= $(HOST_OS)

CUDA		?= /usr/local/cuda
NVCC		= $(CUDA)/bin/nvcc
NVCCFLAGS	= -I$(CUDA)/samples/common/inc/

NVCCFLAGS	+= --use_fast_math
#NVCCFLAGS   += -DDOUBLE_REALS
NVCCFLAGS	+= -DOUTPUT_PPM -DOUTPUT_POINTS
#NVCCFLAGS	+= -DDUMP_POINTS

ifeq ($(TARGET_OS),darwin)
  LIBS		= -Xlinker -framework,GLUT -Xlinker -framework,OpenGL
else
  LIBS += $(GLLINK)
  LIBS += -lGL -lGLU -lX11 -lglut
endif

MAINS		= $(wildcard lyap_*.cu)
SRCS		= $(wildcard *.cu)
BINS		= $(patsubst %.cu,%,$(MAINS))
OBJS		= $(patsubst %.cu,%.o,$(SRCS))

all: $(BINS)



lyap_interactive: lyap_interactive.o kernel.o scene.o params.o
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

lyap_calculate: lyap_calculate.o kernel.o scene.o params.o
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

timeout:
	echo 'N' | sudo tee /sys/kernel/debug/gpu.0/timeouts_enabled

kernel.o: kernel.cu vec3.hpp

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LIBS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -rf $(BINS) $(OBJS) *.dSYM
