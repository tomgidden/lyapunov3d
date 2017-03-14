#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "lyaputils.h"

// Unused
unsigned int windowWidth = 1024;
unsigned int windowHeight = 1024;
unsigned int renderDenominator = 4;

int main(int argc, char** argv)
{
  char *ifn = argv[--argc];

  int ifd;
  if(!(ifd = open(ifn, O_RDONLY))) {
    fprintf(stderr, "Could not open input file\n");
    return 1;
  }

  if(loadLyapHeader(ifd)) {
    fprintf(stderr, "Could not read input file\n");
    return 1;
  }

  dumpLyapHeader(1);

  return 0;
}
