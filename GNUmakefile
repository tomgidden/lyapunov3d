CFLAGS+=-Wall -W -std=gnu99 -g
CFLAGS+=-Wno-deprecated-declarations

#CFLAGS+=-I/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/ -I/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers/
LDFLAGS+=-framework OpenCL -framework GLUT -framework OpenGL

#CFLAGS+=-I/usr/X11/include/libpng15
#LDFLAGS+=-L/usr/X11/lib -lpng
LDFLAGS+=-lpng

all: interactive offline

offline: calculator renderer dumpheader

dumpheader.o: cglutils.h real4.h common.h params.h dumpheader.c

dumpheader: dumpheader.o cglutils.o lyaputils.o

calculator.o: cglutils.h real4.h common.h params.h

calculator: calculator.o cglutils.o lyaputils.o

renderer.o: cglutils.h real4.h common.h params.h

renderer: renderer.o cglutils.o lyaputils.o

interactive.o: cglutils.h real4.h common.h params.h lyaputils.h

interactive: interactive.o cglutils.o lyaputils.o

clean:
	rm -rf interactive calculator renderer dumpheader *.o *.dSYM
