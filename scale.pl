#!/usr/bin/env perl
use Getopt::Std;
getopts('n');

sub easeInOutQuart {
    my ($t, $b, $c, $d) = @_;
    $t /= $d/2;
    return $c/2*$t*$t + $b if ($t < 1) ;
    $t --;
    return -$c/2*($t*($t - 2) - 1) + $b;
}


use Data::Dumper;
for ($t=0; $t<=1.00001; $t += 0.01) {
    $i = easeInOutQuart($t, 0, 1.0, 1.0);

#        $fn = sprintf("Render_*_3840x2160_BCABA_cx=%.8g_cy=%.8g_cz=%.8g_*", $i, $i, $i);
#        @files = glob($fn);
#        next if $#files >= 0;

#        if ($opt_n) {
#            print "$fn\n";
#            next;
#        }


    open IN, "params.cu.dist";
    $buf = join('', <IN>);
    close IN;

    $params = <<"EOF";
    Vec dir = Vec(4,4,4);

    Vec side = Vec(-4,4,4);
    side.normalize();

    Vec up = side * dir.normalized();
    up.normalize();

    Quat rot0 = Quat(up, -20, 1);
    Quat rot1 = Quat(up, 20, 1);

    Quat nrot = rot0.nlerp(rot1, $i);
    Vec nd = nrot.transform(dir).normalized() * -1.0 ;

    cam.C = Vec(4.0 - 0.9*$i, 4.0 - 0.9*$i, 4.0 - 0.9*$i) - nd;
    cam.Q = Quat(Vec(0,0,1), nd, 1.0);

//    cam.C = Vec(4.125, 4.125, 4.125);
//    cam.Q = Quat(0.820473,-0.339851,-0.175920,0.424708);

EOF

    $buf =~ s|(/\*startcalc\*/)(.*)(/\*endcalc\*/)|$1$params$3|s;

    print "$t\t$i\n";

    unless ($opt_n) {
        open OUT, "> params.cu";
        print OUT $buf;
        close OUT;

        @args = ("make", "lyap_interactive", "run");
        system(@args);
    }
}
#Render_1526604923_8192x8192_BCABA_step=2_D=2.1_i=18,1008_d=8192_j=0.5_r=32_ot=-0.75_time=44h54m12s.png
