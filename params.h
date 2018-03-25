LyapParams prm = {
  .d = 2.10,
  .settle = 5,
  .accum = 10,
  .stepMethod = 3,
  .nearThreshold = -1.000000,
  .nearMultiplier = 2.000000,
  .opaqueThreshold = -1.125000,
  .chaosThreshold = 100000.000000,
  .depth = 16.000000,
  .jitter = 0.500000,
  .refine = 32.000000,
  .gradient = 0.010000,
  .lMin = 0.000000,
  .lMax = 4.000000
};

sequence = "AAAAAABBBBBBCCCCCCDDDDDD";

LyapCam cam = {
  .C = {4.010000,4.000000,4.000000,0.000000},
  .Q = {0.820473,-0.339851,-0.175920,0.424708},
  .M = 0.500000
};

Ls[0] = {
  .C = {5.000000,7.000000,3.000000,0.000000},
  .Q = {0.710595,0.282082,-0.512168,0.391368},
  .M = 0.500000,
  .lightInnerCone = 0.904535,
  .lightOuterCone = 0.816497,
  .lightRange = 1.000000,
  .ambient = {0.000000,0.000000,0.000000,0.000000},
  .diffuseColor = {0.300000,0.400000,0.500000,0.000000},
  .diffusePower = 10.000000,
  .specularColor = {0.900000,0.900000,0.900000,0.000000},
  .specularPower = 10.000000,
  .specularHardness = 10.000000,
  .chaosColor = {0.000000,0.000000,0.000000,0.000000}
};

Ls[1] = {
  .C = {3.000000,7.000000,5.000000,0.000000},
  .Q = {0.039640,0.840027,-0.538582,-0.052093},
  .M = 1.677200,
  .lightInnerCone = 0.534489,
  .lightOuterCone = 0.388485,
  .lightRange = 0.500000,
  .ambient = {0.000000,0.000000,0.000000,0.000000},
  .diffuseColor = {0.300000,0.374694,0.200000,0.000000},
  .diffusePower = 10.000000,
  .specularColor = {1.000000,1.000000,1.000000,0.000000},
  .specularPower = 10.000000,
  .specularHardness = 10.000000,
  .chaosColor = {0.000000,0.000000,0.000000,0.000000}
};
