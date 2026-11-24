// =============================================================================
// Fixed-Path TDM Circuit Switch:
//     1. timeslot_expansion_register (Register holding Init Value)
//     2. subslot_down_counter        (Down Counter: Init Value, Init, Zero)
//     3. timeslot_up_counter         (Up Counter: En driven by Zero, Count)
//     4. tdm_schedule_decoder        (Combinational Decoder: Timeslot -> Config)
//     5. switch_input_buffers        (1-Register Input Buffers on 5 input ports)
//     6. switch_crossbar             (Combinational 5x5 Multicast Crossbar)
// =============================================================================
`timescale 1ns/1ps
`default_nettype none

// -----------------------------------------------------------------------------
// 1. Timeslot Expansion Register (Fig. 2b: "Timeslot Expansion Register")
// -----------------------------------------------------------------------------
module timeslot_expansion_register #(
    parameter integer COUNTER_WIDTH = 20,
    parameter integer SUBSLOT_WIDTH = 16,
    parameter integer TAIL_CYCLES   = 7
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     load_en,
    input  wire [SUBSLOT_WIDTH-1:0] cfg_inject_cycles,
    output reg  [SUBSLOT_WIDTH-1:0] inject_cycles_q,
    output wire [COUNTER_WIDTH-1:0] init_value
);
    localparam [COUNTER_WIDTH-1:0] TAIL_VAL = TAIL_CYCLES[COUNTER_WIDTH-1:0];

    wire [SUBSLOT_WIDTH-1:0] sanitized_omega =
        (cfg_inject_cycles == {SUBSLOT_WIDTH{1'b0}})
            ? {{(SUBSLOT_WIDTH-1){1'b0}}, 1'b1}
            : cfg_inject_cycles;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inject_cycles_q <= {{(SUBSLOT_WIDTH-1){1'b0}}, 1'b1};
        end else if (load_en) begin
            inject_cycles_q <= sanitized_omega;
        end
    end

    wire [COUNTER_WIDTH-1:0] omega_ext =
        {{(COUNTER_WIDTH-SUBSLOT_WIDTH){1'b0}}, inject_cycles_q};

    assign init_value = omega_ext + TAIL_VAL - {{(COUNTER_WIDTH-1){1'b0}}, 1'b1};
endmodule

// -----------------------------------------------------------------------------
// 2. Subslot Down Counter (Fig. 2b: "Subslot Counter / Down Counter")
//    - Loads Init Value when Init is high
//    - Asserts Zero when count reaches 0
//    - Zero loops back to Init (Fig. 2b) and drives Timeslot Counter En
// -----------------------------------------------------------------------------
module subslot_down_counter #(
    parameter integer COUNTER_WIDTH = 20
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     running,
    input  wire                     force_init,
    input  wire [COUNTER_WIDTH-1:0] init_value,
    output reg  [COUNTER_WIDTH-1:0] count,
    output wire                     zero,
    output wire                     init
);
    assign zero = running && (count == {COUNTER_WIDTH{1'b0}});
    assign init = force_init || zero;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {COUNTER_WIDTH{1'b0}};
        end else if (init) begin
            count <= init_value;
        end else if (running) begin
            count <= count - {{(COUNTER_WIDTH-1){1'b0}}, 1'b1};
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 3. Timeslot Up Counter (Fig. 2b: "Timeslot Counter / Up Counter")
//    - Enabled by En (wired from Subslot Down Counter's Zero output)
//    - Outputs Timeslot (Count) in 0 .. NUM_TIMESLOTS-1
// -----------------------------------------------------------------------------
module timeslot_up_counter #(
    parameter integer NUM_TIMESLOTS = 64,
    parameter integer TS_WIDTH      = 6
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                clear,
    input  wire                en,
    output reg  [TS_WIDTH-1:0] count,
    output wire                last_timeslot
);
    localparam integer LAST_SLOT_INT = NUM_TIMESLOTS - 1;
    localparam [TS_WIDTH-1:0] LAST_SLOT = LAST_SLOT_INT[TS_WIDTH-1:0];

    assign last_timeslot = (count == LAST_SLOT);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= {TS_WIDTH{1'b0}};
        end else if (clear) begin
            count <= {TS_WIDTH{1'b0}};
        end else if (en) begin
            if (last_timeslot)
                count <= {TS_WIDTH{1'b0}};
            else
                count <= count + {{(TS_WIDTH-1){1'b0}}, 1'b1};
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 4. TDM Schedule Decoder (Fig. 2b: "TDM Schedule Decoder")
//    - Combinational synthesized decoding: In (Timeslot) -> Out (Crossbar Config)
// -----------------------------------------------------------------------------
module tdm_schedule_decoder #(
    parameter integer NUM_TIMESLOTS = 64,
    parameter integer TS_WIDTH      = 6,
    parameter [15*NUM_TIMESLOTS-1:0] SWITCH_CONFIG = {15*NUM_TIMESLOTS{1'b1}}
)(
    input  wire [TS_WIDTH-1:0] in_timeslot,
    output reg  [14:0]         out_crossbar_config
);
    integer idx;
    always @(*) begin
        out_crossbar_config = 15'h7fff;
        for (idx = 0; idx < NUM_TIMESLOTS; idx = idx + 1) begin
            if (in_timeslot == idx[TS_WIDTH-1:0]) begin
                out_crossbar_config = SWITCH_CONFIG[15*idx +: 15];
            end
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 5. Switch Input Buffers (Fig. 2b: "Input Buffers")
//    - Strictly ONE register per input port (In_East, In_West, In_North,
//      In_South, In_Local) placed BEFORE the Crossbar.
// -----------------------------------------------------------------------------
module switch_input_buffers #(
    parameter integer DATA_WIDTH = 32
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  running,
    input  wire [DATA_WIDTH-1:0] in_east_data,
    input  wire                  in_east_valid,
    input  wire [DATA_WIDTH-1:0] in_west_data,
    input  wire                  in_west_valid,
    input  wire [DATA_WIDTH-1:0] in_north_data,
    input  wire                  in_north_valid,
    input  wire [DATA_WIDTH-1:0] in_south_data,
    input  wire                  in_south_valid,
    input  wire [DATA_WIDTH-1:0] in_local_data,
    input  wire                  in_local_valid,
    output reg  [DATA_WIDTH-1:0] buf_east_data,
    output reg                   buf_east_valid,
    output reg  [DATA_WIDTH-1:0] buf_west_data,
    output reg                   buf_west_valid,
    output reg  [DATA_WIDTH-1:0] buf_north_data,
    output reg                   buf_north_valid,
    output reg  [DATA_WIDTH-1:0] buf_south_data,
    output reg                   buf_south_valid,
    output reg  [DATA_WIDTH-1:0] buf_local_data,
    output reg                   buf_local_valid
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_east_data  <= {DATA_WIDTH{1'b0}}; buf_east_valid  <= 1'b0;
            buf_west_data  <= {DATA_WIDTH{1'b0}}; buf_west_valid  <= 1'b0;
            buf_north_data <= {DATA_WIDTH{1'b0}}; buf_north_valid <= 1'b0;
            buf_south_data <= {DATA_WIDTH{1'b0}}; buf_south_valid <= 1'b0;
            buf_local_data <= {DATA_WIDTH{1'b0}}; buf_local_valid <= 1'b0;
        end else if (clear || !running) begin
            buf_east_data  <= {DATA_WIDTH{1'b0}}; buf_east_valid  <= 1'b0;
            buf_west_data  <= {DATA_WIDTH{1'b0}}; buf_west_valid  <= 1'b0;
            buf_north_data <= {DATA_WIDTH{1'b0}}; buf_north_valid <= 1'b0;
            buf_south_data <= {DATA_WIDTH{1'b0}}; buf_south_valid <= 1'b0;
            buf_local_data <= {DATA_WIDTH{1'b0}}; buf_local_valid <= 1'b0;
        end else begin
            buf_east_valid  <= in_east_valid;
            buf_east_data   <= in_east_valid  ? in_east_data  : {DATA_WIDTH{1'b0}};

            buf_west_valid  <= in_west_valid;
            buf_west_data   <= in_west_valid  ? in_west_data  : {DATA_WIDTH{1'b0}};

            buf_north_valid <= in_north_valid;
            buf_north_data  <= in_north_valid ? in_north_data : {DATA_WIDTH{1'b0}};

            buf_south_valid <= in_south_valid;
            buf_south_data  <= in_south_valid ? in_south_data : {DATA_WIDTH{1'b0}};

            buf_local_valid <= in_local_valid;
            buf_local_data  <= in_local_valid ? in_local_data : {DATA_WIDTH{1'b0}};
        end
    end
endmodule

// -----------------------------------------------------------------------------
// 6. Switch Crossbar (Fig. 2b: "Crossbar")
//    - Combinational 1-to-many crossbar routing from the 1-register Input Buffers
//      to Out_East, Out_West, Out_North, Out_South, and Out_Local.
//    - Port selector codes:
//        3'd0 : North
//        3'd1 : South
//        3'd2 : East
//        3'd3 : West
//        3'd4 : Local
//        3'd7 : Disabled
// -----------------------------------------------------------------------------
module switch_crossbar #(
    parameter integer DATA_WIDTH = 32
)(
    input  wire [14:0]           select,
    input  wire [DATA_WIDTH-1:0] in_east_data,
    input  wire                  in_east_valid,
    input  wire [DATA_WIDTH-1:0] in_west_data,
    input  wire                  in_west_valid,
    input  wire [DATA_WIDTH-1:0] in_north_data,
    input  wire                  in_north_valid,
    input  wire [DATA_WIDTH-1:0] in_south_data,
    input  wire                  in_south_valid,
    input  wire [DATA_WIDTH-1:0] in_local_data,
    input  wire                  in_local_valid,
    output wire [DATA_WIDTH-1:0] out_east_data,
    output wire                  out_east_valid,
    output wire [DATA_WIDTH-1:0] out_west_data,
    output wire                  out_west_valid,
    output wire [DATA_WIDTH-1:0] out_north_data,
    output wire                  out_north_valid,
    output wire [DATA_WIDTH-1:0] out_south_data,
    output wire                  out_south_valid,
    output wire [DATA_WIDTH-1:0] out_local_data,
    output wire                  out_local_valid
);
    function [DATA_WIDTH:0] select_input;
        input [2:0]            sel;
        input [DATA_WIDTH-1:0] d_north, d_south, d_east, d_west, d_local;
        input                  v_north, v_south, v_east, v_west, v_local;
        begin
            case (sel)
                3'd0: select_input = {v_north, v_north ? d_north : {DATA_WIDTH{1'b0}}};
                3'd1: select_input = {v_south, v_south ? d_south : {DATA_WIDTH{1'b0}}};
                3'd2: select_input = {v_east,  v_east  ? d_east  : {DATA_WIDTH{1'b0}}};
                3'd3: select_input = {v_west,  v_west  ? d_west  : {DATA_WIDTH{1'b0}}};
                3'd4: select_input = {v_local, v_local ? d_local : {DATA_WIDTH{1'b0}}};
                default: select_input = {(DATA_WIDTH+1){1'b0}};
            endcase
        end
    endfunction

    wire [DATA_WIDTH:0] sel_north = select_input(select[0  +: 3], in_north_data, in_south_data, in_east_data, in_west_data, in_local_data,
                                                                  in_north_valid, in_south_valid, in_east_valid, in_west_valid, in_local_valid);
    wire [DATA_WIDTH:0] sel_south = select_input(select[3  +: 3], in_north_data, in_south_data, in_east_data, in_west_data, in_local_data,
                                                                  in_north_valid, in_south_valid, in_east_valid, in_west_valid, in_local_valid);
    wire [DATA_WIDTH:0] sel_east  = select_input(select[6  +: 3], in_north_data, in_south_data, in_east_data, in_west_data, in_local_data,
                                                                  in_north_valid, in_south_valid, in_east_valid, in_west_valid, in_local_valid);
    wire [DATA_WIDTH:0] sel_west  = select_input(select[9  +: 3], in_north_data, in_south_data, in_east_data, in_west_data, in_local_data,
                                                                  in_north_valid, in_south_valid, in_east_valid, in_west_valid, in_local_valid);
    wire [DATA_WIDTH:0] sel_local = select_input(select[12 +: 3], in_north_data, in_south_data, in_east_data, in_west_data, in_local_data,
                                                                  in_north_valid, in_south_valid, in_east_valid, in_west_valid, in_local_valid);

    assign {out_north_valid, out_north_data} = sel_north;
    assign {out_south_valid, out_south_data} = sel_south;
    assign {out_east_valid,  out_east_data}  = sel_east;
    assign {out_west_valid,  out_west_data}  = sel_west;
    assign {out_local_valid, out_local_data} = sel_local;
endmodule

// -----------------------------------------------------------------------------
// Top-Level Switch: fixed_path_switch_5port (Fig. 2b)
// -----------------------------------------------------------------------------
module fixed_path_switch_5port #(
    parameter integer DATA_WIDTH    = 32,
    parameter integer NUM_TIMESLOTS = 64,
    parameter integer SUBSLOT_WIDTH = 16,
    parameter integer TAIL_CYCLES   = 7,
    parameter [15*NUM_TIMESLOTS-1:0] SWITCH_CONFIG = {15*NUM_TIMESLOTS{1'b1}}
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,
    input  wire [SUBSLOT_WIDTH-1:0] cfg_inject_cycles,

    // Figure 2(b) Input Ports
    input  wire [DATA_WIDTH-1:0]    i_east,
    input  wire                     i_east_valid,
    input  wire [DATA_WIDTH-1:0]    i_west,
    input  wire                     i_west_valid,
    input  wire [DATA_WIDTH-1:0]    i_north,
    input  wire                     i_north_valid,
    input  wire [DATA_WIDTH-1:0]    i_south,
    input  wire                     i_south_valid,
    input  wire [DATA_WIDTH-1:0]    i_local,
    input  wire                     i_local_valid,

    // Figure 2(b) Output Ports
    output wire [DATA_WIDTH-1:0]    o_east,
    output wire                     o_east_valid,
    output wire [DATA_WIDTH-1:0]    o_west,
    output wire                     o_west_valid,
    output wire [DATA_WIDTH-1:0]    o_north,
    output wire                     o_north_valid,
    output wire [DATA_WIDTH-1:0]    o_south,
    output wire                     o_south_valid,
    output wire [DATA_WIDTH-1:0]    o_local,
    output wire                     o_local_valid,

    // Status & Observability Ports
    output reg                      running,
    output wire                     inject_enable,
    output wire [31:0]              timeslot,
    output wire [31:0]              subslot,
    output wire [31:0]              subslot_down_val,
    output wire                     subslot_zero,
    output wire [14:0]              crossbar_config,
    output wire                     slot_last,
    output wire                     frame_last
);
    function integer bits_for;
        input integer count;
        integer v;
        begin
            bits_for = 1;
            for (v = count - 1; v > 1; v = v >> 1)
                bits_for = bits_for + 1;
        end
    endfunction

    localparam integer TS_WIDTH      = bits_for(NUM_TIMESLOTS);
    localparam integer COUNTER_WIDTH = SUBSLOT_WIDTH + bits_for(TAIL_CYCLES + 1) + 1;
    localparam [COUNTER_WIDTH-1:0] TAIL_VAL = TAIL_CYCLES[COUNTER_WIDTH-1:0];

    wire [SUBSLOT_WIDTH-1:0] inject_cycles_q;
    wire [COUNTER_WIDTH-1:0] exp_init_value;
    wire [COUNTER_WIDTH-1:0] down_count;
    wire                     down_zero;
    wire                     down_init;
    wire [TS_WIDTH-1:0]      ts_count;
    wire                     ts_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            running <= 1'b0;
        else if (start)
            running <= 1'b1;
    end

    wire [SUBSLOT_WIDTH-1:0] start_omega =
        (cfg_inject_cycles == {SUBSLOT_WIDTH{1'b0}})
            ? {{(SUBSLOT_WIDTH-1){1'b0}}, 1'b1}
            : cfg_inject_cycles;

    wire [COUNTER_WIDTH-1:0] effective_init_val =
        start ? ({{(COUNTER_WIDTH-SUBSLOT_WIDTH){1'b0}}, start_omega} + TAIL_VAL - {{(COUNTER_WIDTH-1){1'b0}}, 1'b1})
              : exp_init_value;

    // 1. Timeslot Expansion Register (Fig. 2b)
    timeslot_expansion_register #(
        .COUNTER_WIDTH(COUNTER_WIDTH),
        .SUBSLOT_WIDTH(SUBSLOT_WIDTH),
        .TAIL_CYCLES  (TAIL_CYCLES)
    ) u_timeslot_expansion_reg (
        .clk              (clk),
        .rst_n            (rst_n),
        .load_en          (start),
        .cfg_inject_cycles(cfg_inject_cycles),
        .inject_cycles_q  (inject_cycles_q),
        .init_value       (exp_init_value)
    );

    // 2. Subslot Down Counter (Fig. 2b)
    subslot_down_counter #(
        .COUNTER_WIDTH(COUNTER_WIDTH)
    ) u_subslot_down_counter (
        .clk       (clk),
        .rst_n     (rst_n),
        .running   (running && !start),
        .force_init(start),
        .init_value(effective_init_val),
        .count     (down_count),
        .zero      (down_zero),
        .init      (down_init)
    );

    // 3. Timeslot Up Counter (Fig. 2b)
    timeslot_up_counter #(
        .NUM_TIMESLOTS(NUM_TIMESLOTS),
        .TS_WIDTH     (TS_WIDTH)
    ) u_timeslot_up_counter (
        .clk          (clk),
        .rst_n        (rst_n),
        .clear        (start),
        .en           (down_zero),
        .count        (ts_count),
        .last_timeslot(ts_last)
    );

    // 4. TDM Schedule Decoder (Fig. 2b)
    tdm_schedule_decoder #(
        .NUM_TIMESLOTS(NUM_TIMESLOTS),
        .TS_WIDTH     (TS_WIDTH),
        .SWITCH_CONFIG(SWITCH_CONFIG)
    ) u_tdm_schedule_decoder (
        .in_timeslot        (ts_count),
        .out_crossbar_config(crossbar_config)
    );

    // 5. 1-Register Input Buffers (Fig. 2b: Input Buffers before Crossbar)
    wire [DATA_WIDTH-1:0] buf_east_d,  buf_west_d,  buf_north_d,  buf_south_d,  buf_local_d;
    wire                  buf_east_v,  buf_west_v,  buf_north_v,  buf_south_v,  buf_local_v;

    switch_input_buffers #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_input_buffers (
        .clk           (clk),
        .rst_n         (rst_n),
        .clear         (start),
        .running       (running && !start),
        .in_east_data  (i_east),
        .in_east_valid (i_east_valid),
        .in_west_data  (i_west),
        .in_west_valid (i_west_valid),
        .in_north_data (i_north),
        .in_north_valid(i_north_valid),
        .in_south_data (i_south),
        .in_south_valid(i_south_valid),
        .in_local_data (i_local),
        .in_local_valid(i_local_valid && inject_enable),
        .buf_east_data (buf_east_d),
        .buf_east_valid(buf_east_v),
        .buf_west_data (buf_west_d),
        .buf_west_valid(buf_west_v),
        .buf_north_data(buf_north_d),
        .buf_north_valid(buf_north_v),
        .buf_south_data(buf_south_d),
        .buf_south_valid(buf_south_v),
        .buf_local_data(buf_local_d),
        .buf_local_valid(buf_local_v)
    );

    // 6. Crossbar (Fig. 2b)
    switch_crossbar #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_crossbar (
        .select         (crossbar_config),
        .in_east_data   (buf_east_d),
        .in_east_valid  (buf_east_v),
        .in_west_data   (buf_west_d),
        .in_west_valid  (buf_west_v),
        .in_north_data  (buf_north_d),
        .in_north_valid (buf_north_v),
        .in_south_data  (buf_south_d),
        .in_south_valid (buf_south_v),
        .in_local_data  (buf_local_d),
        .in_local_valid (buf_local_v),
        .out_east_data  (o_east),
        .out_east_valid (o_east_valid),
        .out_west_data  (o_west),
        .out_west_valid (o_west_valid),
        .out_north_data (o_north),
        .out_north_valid(o_north_valid),
        .out_south_data (o_south),
        .out_south_valid(o_south_valid),
        .out_local_data (o_local),
        .out_local_valid(o_local_valid)
    );

    wire [COUNTER_WIDTH-1:0] subslot_asc = running ? (exp_init_value - down_count) : {COUNTER_WIDTH{1'b0}};

    assign inject_enable    = running && !start && (down_count >= TAIL_VAL);
    assign subslot_zero     = down_zero;
    assign slot_last        = down_zero;
    assign frame_last       = down_zero && ts_last;
    assign timeslot         = {{(32-TS_WIDTH){1'b0}}, ts_count};
    assign subslot          = {{(32-COUNTER_WIDTH){1'b0}}, subslot_asc};
    assign subslot_down_val = {{(32-COUNTER_WIDTH){1'b0}}, down_count};

endmodule
`default_nettype wire