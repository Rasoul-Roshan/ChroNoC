// Five-port fixed-path TDM circuit switch.
`timescale 1ns/1ps
`default_nettype none
module fixed_path_switch_5port #(
    parameter integer DATA_WIDTH = 32,
    parameter integer NUM_TIMESLOTS = 64,
    parameter integer SUBSLOT_WIDTH = 16,
    parameter integer TAIL_CYCLES = 7,

    parameter [15*NUM_TIMESLOTS-1:0] SWITCH_CONFIG = {15*NUM_TIMESLOTS{1'b1}}
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [SUBSLOT_WIDTH-1:0] cfg_inject_cycles,
    input wire [DATA_WIDTH-1:0] i_north,
    input wire i_north_valid,
    input wire [DATA_WIDTH-1:0] i_south,
    input wire i_south_valid,
    input wire [DATA_WIDTH-1:0] i_east,
    input wire i_east_valid,
    input wire [DATA_WIDTH-1:0] i_west,
    input wire i_west_valid,
    input wire [DATA_WIDTH-1:0] i_local,
    input wire i_local_valid,
    output reg [DATA_WIDTH-1:0] o_north,
    output reg o_north_valid,
    output reg [DATA_WIDTH-1:0] o_south,
    output reg o_south_valid,
    output reg [DATA_WIDTH-1:0] o_east,
    output reg o_east_valid,
    output reg [DATA_WIDTH-1:0] o_west,
    output reg o_west_valid,
    output reg [DATA_WIDTH-1:0] o_local,
    output reg o_local_valid,
    output reg running,
    output wire inject_enable,
    output wire [31:0] timeslot,
    output wire [31:0] subslot,
    output wire slot_last,
    output wire frame_last
);
    function integer bits_for;
        input integer count;
        integer v;
        begin
            bits_for = 1;
            for (v=count-1; v>1; v=v>>1) bits_for=bits_for+1;
        end
    endfunction
    localparam integer TS_WIDTH = bits_for(NUM_TIMESLOTS);
     
    localparam integer PHASE_WIDTH = SUBSLOT_WIDTH + bits_for(TAIL_CYCLES+1) + 1;
    localparam [PHASE_WIDTH-1:0] TAIL_VALUE = TAIL_CYCLES[PHASE_WIDTH-1:0];
    localparam integer LAST_SLOT_INT = NUM_TIMESLOTS-1;
    localparam [TS_WIDTH-1:0] LAST_SLOT = LAST_SLOT_INT[TS_WIDTH-1:0];
    reg [TS_WIDTH-1:0] slot_q;
    reg [PHASE_WIDTH-1:0] phase_q;
    reg [SUBSLOT_WIDTH-1:0] inject_cycles_q;
    wire [PHASE_WIDTH-1:0] slot_cycles;
    wire [PHASE_WIDTH-1:0] inject_limit;
    wire [14:0] control;
    assign inject_limit = {{(PHASE_WIDTH-SUBSLOT_WIDTH){1'b0}},inject_cycles_q};
    assign slot_cycles = inject_limit + TAIL_VALUE;
    assign timeslot = {{(32-TS_WIDTH){1'b0}},slot_q};
    assign subslot = {{(32-PHASE_WIDTH){1'b0}},phase_q};
    assign control = SWITCH_CONFIG[15*slot_q +: 15];
    assign inject_enable = running && !start && (phase_q < inject_limit);
    assign slot_last = running && !start && (phase_q == slot_cycles-1'b1);
    assign frame_last = slot_last && (slot_q == LAST_SLOT);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            slot_q <= 0;
            phase_q <= 0;
            inject_cycles_q <= 1;
        end else if (start) begin
            running <= 1'b1;
            slot_q <= 0;
            phase_q <= 0;
            inject_cycles_q <= (cfg_inject_cycles == 0) ? 1 : cfg_inject_cycles;
        end else if (running) begin
            if (slot_last) begin
                phase_q <= 0;
                if (slot_q == LAST_SLOT) slot_q <= 0;
                else slot_q <= slot_q + 1'b1;
            end else phase_q <= phase_q + 1'b1;
        end
    end

    function [DATA_WIDTH:0] select_input;
        input [2:0] sel;
        begin
            case (sel)
                3'd0: select_input = {i_north_valid, i_north};
                3'd1: select_input = {i_south_valid, i_south};
                3'd2: select_input = {i_east_valid, i_east};
                3'd3: select_input = {i_west_valid, i_west};
                3'd4: select_input = {(i_local_valid && inject_enable), i_local};
                default: select_input = {(DATA_WIDTH+1){1'b0}};
            endcase
        end
    endfunction
    wire [DATA_WIDTH:0] selected_north;
    assign selected_north = select_input(control[0 +: 3]);
    wire [DATA_WIDTH:0] selected_south;
    assign selected_south = select_input(control[3 +: 3]);
    wire [DATA_WIDTH:0] selected_east;
    assign selected_east = select_input(control[6 +: 3]);
    wire [DATA_WIDTH:0] selected_west;
    assign selected_west = select_input(control[9 +: 3]);
    wire [DATA_WIDTH:0] selected_local;
    assign selected_local = select_input(control[12 +: 3]);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_north <= 0; o_north_valid <= 1'b0;
            o_south <= 0; o_south_valid <= 1'b0;
            o_east <= 0; o_east_valid <= 1'b0;
            o_west <= 0; o_west_valid <= 1'b0;
            o_local <= 0; o_local_valid <= 1'b0;
        end else if (start || !running) begin
            o_north <= 0; o_north_valid <= 1'b0;
            o_south <= 0; o_south_valid <= 1'b0;
            o_east <= 0; o_east_valid <= 1'b0;
            o_west <= 0; o_west_valid <= 1'b0;
            o_local <= 0; o_local_valid <= 1'b0;
        end else begin
            o_north_valid <= selected_north[DATA_WIDTH];
            o_north <= selected_north[DATA_WIDTH] ? selected_north[DATA_WIDTH-1:0] : {DATA_WIDTH{1'b0}};
            o_south_valid <= selected_south[DATA_WIDTH];
            o_south <= selected_south[DATA_WIDTH] ? selected_south[DATA_WIDTH-1:0] : {DATA_WIDTH{1'b0}};
            o_east_valid <= selected_east[DATA_WIDTH];
            o_east <= selected_east[DATA_WIDTH] ? selected_east[DATA_WIDTH-1:0] : {DATA_WIDTH{1'b0}};
            o_west_valid <= selected_west[DATA_WIDTH];
            o_west <= selected_west[DATA_WIDTH] ? selected_west[DATA_WIDTH-1:0] : {DATA_WIDTH{1'b0}};
            o_local_valid <= selected_local[DATA_WIDTH];
            o_local <= selected_local[DATA_WIDTH] ? selected_local[DATA_WIDTH-1:0] : {DATA_WIDTH{1'b0}};
        end
    end
endmodule
`default_nettype wire
