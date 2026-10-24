// Fixed-path 4x4 TDM mesh.
`timescale 1ns/1ps
`default_nettype none
module fixed_path_noc_mesh_4x4_t64 #(
    parameter integer DATA_WIDTH = 32,
    parameter integer NUM_TIMESLOTS = 64,
    parameter integer SUBSLOT_WIDTH = 16,
     
    parameter integer TAIL_CYCLES = 7
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [SUBSLOT_WIDTH-1:0] cfg_inject_cycles,
    input wire [16*DATA_WIDTH-1:0] i_local_flat,
    input wire [15:0] i_local_valid,
    output wire [16*DATA_WIDTH-1:0] o_local_flat,
    output wire [15:0] o_local_valid,
    output wire [15:0] o_local_ready,
    output wire running,
    output wire [31:0] timeslot,
    output wire [31:0] subslot,
    output wire slot_last,
    output wire frame_last
);

    function [15*NUM_TIMESLOTS-1:0] make_config;
        input integer node;
        integer t, root, x, y, sx, sy;
        reg [2:0] parent;
        reg [14:0] c;
        begin
            make_config = {15*NUM_TIMESLOTS{1'b1}};
            x = node % 4; y = node / 4;
            for (t=0; t<NUM_TIMESLOTS; t=t+1) begin
                root=t%16; sx=root%4; sy=root/4;
                if (node==root) parent=3'd4;
                else if (y<sy) parent=3'd1;
                else if (y>sy) parent=3'd0;
                else if (x<sx) parent=3'd2;
                else parent=3'd3;
                c=15'h7fff;
                if (y<=sy && y>0) c[2:0]=parent;
                if (y>=sy && y<3) c[5:3]=parent;
                if (y==sy && x>=sx && x<3) c[8:6]=parent;
                if (y==sy && x<=sx && x>0) c[11:9]=parent;
                if (node!=root) c[14:12]=parent;
                make_config[15*t +: 15]=c;
            end
        end
    endfunction
    wire [16*DATA_WIDTH-1:0] n_data, s_data, e_data, w_data;
    wire [15:0] n_valid, s_valid, e_valid, w_valid;
    wire [15:0] sw_running, sw_inject, sw_slot_last, sw_frame_last;
    wire [16*32-1:0] sw_slot, sw_subslot;
    assign running=sw_running[0];
    assign timeslot=sw_slot[0 +: 32];
    assign subslot=sw_subslot[0 +: 32];
    assign slot_last=sw_slot_last[0];
    assign frame_last=sw_frame_last[0];
    genvar g;
    generate for (g=0; g<16; g=g+1) begin: nodes
        wire [DATA_WIDTH-1:0] ni, si, ei, wi;
        wire nv, sv, ev, wv;
        if (g>=4) begin: north_link
            assign ni=s_data[(g-4)*DATA_WIDTH +: DATA_WIDTH];
            assign nv=s_valid[g-4];
        end else begin: north_edge
            assign ni={DATA_WIDTH{1'b0}}; assign nv=1'b0;
        end
        if (g<12) begin: south_link
            assign si=n_data[(g+4)*DATA_WIDTH +: DATA_WIDTH];
            assign sv=n_valid[g+4];
        end else begin: south_edge
            assign si={DATA_WIDTH{1'b0}}; assign sv=1'b0;
        end
        if ((g%4)<3) begin: east_link
            assign ei=w_data[(g+1)*DATA_WIDTH +: DATA_WIDTH];
            assign ev=w_valid[g+1];
        end else begin: east_edge
            assign ei={DATA_WIDTH{1'b0}}; assign ev=1'b0;
        end
        if ((g%4)>0) begin: west_link
            assign wi=e_data[(g-1)*DATA_WIDTH +: DATA_WIDTH];
            assign wv=e_valid[g-1];
        end else begin: west_edge
            assign wi={DATA_WIDTH{1'b0}}; assign wv=1'b0;
        end
        assign o_local_ready[g]=sw_inject[g] && ((timeslot%16)==g);
        fixed_path_switch_5port #(
            .DATA_WIDTH(DATA_WIDTH), .NUM_TIMESLOTS(NUM_TIMESLOTS),
            .SUBSLOT_WIDTH(SUBSLOT_WIDTH), .TAIL_CYCLES(TAIL_CYCLES),
            .SWITCH_CONFIG(make_config(g))
        ) sw (
            .clk(clk), .rst_n(rst_n), .start(start),
            .cfg_inject_cycles(cfg_inject_cycles),
            .i_north(ni), .i_north_valid(nv),
            .i_south(si), .i_south_valid(sv),
            .i_east(ei), .i_east_valid(ev),
            .i_west(wi), .i_west_valid(wv),
            .i_local(i_local_flat[g*DATA_WIDTH +: DATA_WIDTH]),
            .i_local_valid(i_local_valid[g] && o_local_ready[g]),
            .o_north(n_data[g*DATA_WIDTH +: DATA_WIDTH]), .o_north_valid(n_valid[g]),
            .o_south(s_data[g*DATA_WIDTH +: DATA_WIDTH]), .o_south_valid(s_valid[g]),
            .o_east(e_data[g*DATA_WIDTH +: DATA_WIDTH]), .o_east_valid(e_valid[g]),
            .o_west(w_data[g*DATA_WIDTH +: DATA_WIDTH]), .o_west_valid(w_valid[g]),
            .o_local(o_local_flat[g*DATA_WIDTH +: DATA_WIDTH]),
            .o_local_valid(o_local_valid[g]), .running(sw_running[g]),
            .inject_enable(sw_inject[g]), .timeslot(sw_slot[g*32 +: 32]),
            .subslot(sw_subslot[g*32 +: 32]), .slot_last(sw_slot_last[g]),
            .frame_last(sw_frame_last[g])
        );
    end endgenerate
endmodule
`default_nettype wire
