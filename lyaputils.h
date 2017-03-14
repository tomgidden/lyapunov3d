#include "common.h"

#define MAX_LIGHTS 10

extern LyapCamLight Ls[];
extern unsigned int numLights;
extern LyapCamLight cam;
extern LyapParams prm;

extern unsigned int windowWidth;
extern unsigned int windowHeight;
extern unsigned int renderDenominator;

extern LyapCamLight *curC;
extern LyapParams *curP;

extern char *sequence;

extern int loadLyapHeader(int fd);
extern int loadLyapData(int fd, LyapPoint **dataP, size_t *dataSizeP);
extern void dumpLyapHeader(int fd);
extern int saveLyapHeader(int fd);
extern off_t saveLyapBlock(char *fn, off_t offset, LyapPoint *data, size_t dataSize);
extern void camCalculate (LyapCamLight *cam, UINT tw, UINT th, UINT td);
