typedef struct _LyapCamLight {
  REAL4 C;
  QUAT Q;
  REAL M;
  REAL4 V;
  REAL4 S0;
  REAL4 SDX;
  REAL4 SDY;
  UINT textureWidth;
  UINT textureHeight;
  UINT renderWidth;
  UINT renderHeight;
  UINT renderDenominator;

  REAL lightInnerCone, lightOuterCone;
  REAL lightRange;
  REAL4 ambient;
  REAL4 diffuseColor;
  REAL diffusePower;
  REAL4 specularColor;
  REAL specularPower;
  REAL specularHardness;
  REAL4 chaosColor;
} LyapLight;

typedef struct _LyapCamLight LyapCam;

typedef struct {
  REAL d;
  UINT settle;
  UINT accum;
  UINT stepMethod;
  REAL nearThreshold;
  REAL nearMultiplier;
  REAL opaqueThreshold;
  REAL chaosThreshold;
  REAL depth;
  REAL jitter;
  REAL refine;
  REAL gradient;
  REAL lMin;
  REAL lMax;
} LyapParams;

typedef struct {
  REAL x, y, z;
  REAL nx, ny, nz;
  REAL a;
  REAL c;
} LyapPoint;

typedef struct {
    unsigned char r, g, b, a;
} RGBA;
