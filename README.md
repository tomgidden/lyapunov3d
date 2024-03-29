Also: quick-and-dirty 2D Lyapunov in the Terminal window:  https://gist.github.com/tomgidden/1d8145ed8576c99f70975d930895ca6a

Multi-dimensional Lyapunov Fractals
-----------------------------------

This code accompanies the article [Exploring Lyapunov Space](https://gidden.net/2017/03/09/exploring-lyapunov-space/).

As the article explains, I worked on this code initially in 2011, managing
to produce some interesting results. However, I got stuck in the mire of
OpenGL and OpenCL support in OS X and Linux, rather than working on the
algorithm itself -- the interesting bit.

As of March 2018, I've started playing with it again, clearing it up,
and converting from OpenCL to CUDA, and with the aim of getting it 
working well on a Jetson TX2 to allow more experimentation.

I think my goal should be to separate the calculation code completely into a
networked worker, and an entirely separate GUI client that just consumes
data produced by the worker(s); perhaps with a central manager node to
mediate between the two.  Perhaps things like MPI would be a good way to
do this.

## IMPORTANT

**This code monopolises the primary GPU, so your computer may become
completely unresponsive, even once the program has quit.** Yes, even if
you SSH in and kill the program. So, **make sure you have saved all work
and shut down all other apps to minimise the damage of a hard reboot.**

Also, this code is basically seven years old (at the time of writing) and
horribly unfinished and hacky. In particular, the `offline` code is
particularly ropey, as I stopped updating it once I'd started on the
interactive version.

## Installation

This code did once work on an Ubuntu Linux machine, but hasn't been tested
since.  It does seem to work (to some extent) on my 2016 MacBook running
macOS Sierra with only tiny tweaks, which is a minor miracle.

To install on macOS, make sure you have _Homebrew_ installed and do:
```
	brew install libpng glew
```

to install the `glew` OpenGL library used by the interactive app, and the
`png` library used by the offline programs.

Then:
```
    make lyap_interactive && ./lyap_interactive
```

and hope for the best. **I do not accept responsibility for data loss
and/or damage caused by this code.**

Please don't judge me for the poor quality of code. It really doesn't live
up to my personal quality standards -- especially w.r.t. commenting --
which is something I'm really regretting as I pick it up six years later.

## License

Hmm. I suppose it's GPLv3 really. I usually prefer Apache or similar for
my code, but being somewhat of a passion of mine over many years and of
purely academic interest, this work is something I wouldn't mind some
attribution on.

I really do encourage interested parties to help tidy it up and take it
forward, though.  I just don't want to be lost in the kerfuffle.

So unless you arrange with me _in advance_ for it to be licensed
differently, the code is **Copyright 2011-2018, Tom Gidden** and available
under the terms of the GNU Public License version 3.
