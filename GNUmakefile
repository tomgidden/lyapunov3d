CUDA		?= /usr/local/cuda
NVCC		= $(CUDA)/bin/nvcc
NVCCFLAGS	= -I$(CUDA)/samples/common/inc/
NVCCFLAGS	+= --use_fast_math

ifeq ($(TARGET_OS),darwin)
  LIBS		= -Xlinker -framework,GLUT -Xlinker -framework,OpenGL
else
  LIBS += $(GLLINK)
  LIBS += -lGL -lGLU -lX11 -lglut
endif

SRCS		= $(wildcard *.cu)
BINS		= $(patsubst %.cu,%,$(SRCS))
OBJS		= $(patsubst %.cu,%.o,$(SRCS))

all: lyap_interactive

lyap_interactive: lyap_interactive.o lyap.o scene.o params.o
	$(NVCC) $(NVCCFLAGS) $(LIBS) $^ -o $@

timeout:
	echo 'N' | sudo tee /sys/kernel/debug/gpu.0/timeouts_enabled

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LIBS) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -rf $(BINS) $(OBJS) *.dSYM
