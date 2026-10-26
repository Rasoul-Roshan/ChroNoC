# Hardware

Synthesizable Verilog-2001 for the fixed-path TDM switch and 4x4, 64-slot reference
mesh; SystemVerilog testbenches provide self-checking smoke tests.


## Physical implementation results

The [physical results](physical_results/) directory contains the 100 MHz
SKY130A characterization, per-switch power reports, and a post-route layout view.

## Switch timeslot scalability

See [timeslot_scalability](timeslot_scalability/) for post-synthesis area and
VCD-based power results for the decision unit and complete switch at 16-256
timeslots.
