#include "lyaputils.h"
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

LyapCamLight Ls[MAX_LIGHTS];
unsigned int numLights = 0;
LyapCamLight cam;
LyapParams prm;

LyapCamLight *curC = &cam;
LyapParams *curP = &prm;

char *sequence;

void camCalculate (LyapCamLight *camP, UINT tw, UINT th, UINT td)
{
  if(camP->M<QUAT_EPSILON)
    camP->M = QUAT_EPSILON;

  camP->Q = QUAT_normalize(camP->Q);

  if(td>0)
    camP->renderDenominator = td;

  if(tw>0) {
    camP->textureWidth = tw;
    camP->renderWidth = camP->textureWidth/camP->renderDenominator;
  }

  if(th>0) {
    camP->textureHeight = th;
    camP->renderHeight = camP->textureHeight/camP->renderDenominator;
  }

  camP->V = REAL4_normalize(QUAT_transformREAL4(REAL4_init(0,0,1), camP->Q));

  camP->S0 = QUAT_transformREAL4(REAL4_init(-camP->M, -camP->M, 1), camP->Q);

  camP->lightInnerCone = REAL4_dot(camP->V, REAL4_normalize(QUAT_transformREAL4(REAL4_init(-camP->M, -camP->M, 1.5), camP->Q)));
  camP->lightOuterCone = REAL4_dot(camP->V, REAL4_normalize(QUAT_transformREAL4(REAL4_init(-camP->M, -camP->M, 1), camP->Q)));

  camP->SDX = QUAT_transformREAL4(REAL4_init(2*camP->M/(REAL)camP->renderWidth, 0, 0), camP->Q);
  camP->SDY = QUAT_transformREAL4(REAL4_init(0, 2*camP->M/(REAL)camP->renderHeight, 0), camP->Q);
}

int loadLyapHeader(int fd)
{
  size_t i = 0;
  unsigned char buf[1024];
  unsigned char *ptr = buf;
  do {
    if(!(read(fd, ptr, 1)))
      return 1;
    i++;
  }
  while(*ptr++);

  if(!(sequence = malloc(i)))
    return 1;

  memcpy(sequence, buf, i);

  if(sizeof(prm) != read(fd, &prm, sizeof(prm)))
    return 1;

  if(sizeof(cam) != read(fd, &cam, sizeof(cam)))
    return 1;

  if(sizeof(numLights) != read(fd, &numLights, sizeof(numLights)))
    return 1;

  for(i=0; i<numLights; i++)
    if(sizeof(Ls[i]) != read(fd, &Ls[i], sizeof(Ls[i])))
      return 1;

  return 0;
}

int loadLyapData(int fd, LyapPoint **dataP, size_t *dataSizeP)
{
  *dataSizeP = cam.renderWidth * cam.renderHeight * sizeof(LyapPoint);

  if(!(*dataP = malloc(*dataSizeP)))
    return 1;

  read(fd, *dataP, *dataSizeP);

  return 0;
}

int saveLyapHeader(int fd)
{
  int len = strlen(sequence)+1;

  if(len != write(fd, sequence, len))
    return 1;

  if(sizeof(*curP) != write(fd, curP, sizeof(*curP)))
    return 1;

  if(sizeof(*curC) != write(fd, curC, sizeof(*curC)))
    return 1;

  if(sizeof(numLights) != write(fd, &numLights, sizeof(numLights)))
    return 1;

  size_t i;
  for(i=0; i<numLights; i++)
    if(sizeof(Ls[i]) != write(fd, &Ls[i], sizeof(Ls[i])))
      return 1;

  return 0;
}

off_t saveLyapBlock(char *fn, off_t offset, LyapPoint *data, size_t dataSize)
{
  int fd;
  if(!(fd = open(fn, O_APPEND|O_WRONLY))) return 0;

  if(!offset)
    lseek(fd, 0, SEEK_END);
  else
    lseek(fd, offset, SEEK_SET);

  if(dataSize) {
    if((int)dataSize != write(fd, data, dataSize))
      return 0;
  }

  offset = lseek(fd, 0, SEEK_END);

  close(fd);

  return offset;
}

void dumpLyapHeader(int fd)
{
  dprintf(fd,
          "LyapParams __P = {\n"                \
          "  .d = %.2f,\n"                      \
          "  .settle = %d,\n"                   \
          "  .accum = %d,\n"                    \
          "  .stepMethod = %d,\n"               \
          "  .nearThreshold = %f,\n"            \
          "  .nearMultiplier = %f,\n"           \
          "  .opaqueThreshold = %f,\n"          \
          "  .chaosThreshold = %f,\n"           \
          "  .depth = %f,\n"                    \
          "  .jitter = %f,\n"                   \
          "  .refine = %f,\n"                   \
          "  .gradient = %f,\n"                 \
          "  .lMin = %f,\n"                     \
          "  .lMax = %f\n"                      \
          "};\nprm = __P;\n\n",
          prm.d, prm.settle, prm.accum,
          prm.stepMethod,
          prm.nearThreshold,
          prm.nearMultiplier,
          prm.opaqueThreshold,
          prm.chaosThreshold,
          prm.depth,
          prm.jitter,
          prm.refine,
          prm.gradient,
          prm.lMin,
          prm.lMax);

  dprintf(fd, "sequence = \"%s\";\n\n", sequence);

  dprintf(fd,
          "windowWidth = %d;\nwindowHeight = %d;\n\n",
          windowWidth,
          windowHeight);

  dprintf(fd,
          "LyapCamLight __C = {\n"              \
          "  .C = {%f,%f,%f,%f},\n"             \
          "  .Q = {%f,%f,%f,%f},\n"             \
          "  .M = %f\n"                         \
          "};\ncam = __C;\n\n",
          cam.C.x, cam.C.y, cam.C.z, cam.C.w,
          cam.Q.x, cam.Q.y, cam.Q.z, cam.Q.w,
          cam.M);

  unsigned int i;
  for(i=0; i<numLights; i++) {
    dprintf(fd,
            "LyapCamLight __L%d = {\n"                     \
            "  .C = {%f,%f,%f,%f},\n"                      \
            "  .Q = {%f,%f,%f,%f},\n"                      \
            "  .M = %f,\n"                                 \
            "  .lightInnerCone = %f,\n"                    \
            "  .lightOuterCone = %f,\n"                    \
            "  .lightRange = %f,\n"                        \
            "  .ambient = {%f,%f,%f,%f},\n"                \
            "  .diffuseColor = {%f,%f,%f,%f},\n"           \
            "  .diffusePower = %f,\n"                      \
            "  .specularColor = {%f,%f,%f,%f},\n"          \
            "  .specularPower = %f,\n"                     \
            "  .specularHardness = %f,\n"                  \
            "  .chaosColor = {%f,%f,%f,%f}\n"              \
            "};\n"                                         \
            "Ls[%d] = __L%d;\n\n",
            i,
            Ls[i].C.x, Ls[i].C.y, Ls[i].C.z, Ls[i].C.w,
            Ls[i].Q.x, Ls[i].Q.y, Ls[i].Q.z, Ls[i].Q.w,
            Ls[i].M,
            Ls[i].lightInnerCone,
            Ls[i].lightOuterCone,
            Ls[i].lightRange,
            Ls[i].ambient.x,
            Ls[i].ambient.y,
            Ls[i].ambient.z,
            Ls[i].ambient.w,
            Ls[i].diffuseColor.x,
            Ls[i].diffuseColor.y,
            Ls[i].diffuseColor.z,
            Ls[i].diffuseColor.w,
            Ls[i].diffusePower,
            Ls[i].specularColor.x,
            Ls[i].specularColor.y,
            Ls[i].specularColor.z,
            Ls[i].specularColor.w,
            Ls[i].specularPower,
            Ls[i].specularHardness,
            Ls[i].chaosColor.x,
            Ls[i].chaosColor.y,
            Ls[i].chaosColor.z,
            Ls[i].chaosColor.w,
            i, i);
  }
  dprintf(fd, "numLights = %d;\n", numLights);
}
