# Timeslot scalability

Post-synthesis characterization of the isolated combinational route-decision
unit and the complete five-port switch for 16, 32, 64, 128 and 256 timeslots.
Both studies use prefixes of the same nonrepeating configuration sequence.

| Condition | Value |
|---|---|
| Process / library | SKY130A / sky130_fd_sc_hd |
| PVT | TT, 25 C, 1.8 V |
| Full-switch datapath / clock | 32 bits / 100 MHz |
| Slot update rate | 10 MHz |
| Measured window per case | 2048 slots, 204.8 us |
| Power activity | Functional gate-level VCD; 100% reported pin annotation |

- `decision_logic/` and `full_switch/`: summaries and five per-case result folders.
- Each case contains mapped-cell statistics, annotated power, timing, activity
  coverage, RTL and gate-level simulation logs.
- `metadata/`: analysis settings.
- `figures/`: the relative-power figure and its numeric data. 

