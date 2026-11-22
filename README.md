# Strong chromatic index certificates

These files verify the $1.6388\Delta^2$ bound using only the standard library
of Python 3.10 or later. Extract the archive and run this command from the
extracted directory.

~~~sh
python3 -B verify_all.py
~~~

Use `--appendix A`, `--appendix B`, or `--appendix C` to check one appendix.
Use `--jobs 1` to run the palette check in one process. The palette check verifies 487 phases
and the completion step. It ends with the integer palette bound
819392866996 at scale 1000000000000.

The optional `--cpp` flag replays the C++ search in `palette.cpp` that
generated `palette_bounds.txt` and compares every regenerated row with the
recorded file. It requires GNU `g++` with C++20, OpenMP, and signed 128-bit
integer support. To check Appendix C and replay the search, run

~~~sh
python3 -B verify_all.py --appendix C --cpp
~~~

The local flag bases and coefficient data are from
[Davey, Hurley, de Joannis de Verclos, Kang, and Volec's implementation](https://github.com/rossjkang/localflagalgebras/tree/3a429d27d1c6a365f985f2342e43353c934b6563).
See UPSTREAM_LICENSE.txt.
