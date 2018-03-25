#CFLAGS+=-Wall -W -std=gnu99 -g
#CFLAGS+=-Wno-deprecated-declarations

#CFLAGS+=-I/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/ -I/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers/
#LDFLAGS+=-framework OpenCL -framework GLUT -framework OpenGL

#CFLAGS+=-I/Developer/NVIDIA/CUDA-9.1/samples/common/inc/
#CFLAGS+=-I/usr/X11/include/libpng15
#LDFLAGS+=-L/usr/X11/lib -lpng
#LDFLAGS+=-lpng


CUDA		?= /usr/local/cuda
NVCC		= $(CUDA)/bin/nvcc
NVCCFLAGS 	= --use_fast_math -I$(CUDA)/samples/common/inc/
LIBS		= -Xlinker -framework,GLUT -Xlinker -framework,OpenGL
SRCS 		= $(wildcard *.cu)
BINS 		= $(patsubst %.cu,%,$(SRCS))

all: lyap_interactive
#all: $(BINS)

%: %.cu real4.h
	$(NVCC) $(NVCCFLAGS) $(LIBS) $< -o $@

clean:
	rm -rf $(BINS) *.dSYM
