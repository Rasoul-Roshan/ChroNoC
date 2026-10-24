// Self-checking smoke test for the switch.
`timescale 1ns/1ps
module tb_fixed_path_switch_5port;
    localparam W=32, T=3, G=2;
     
    localparam [44:0] CONFIG = {
        3'd7,3'd6,3'd5,3'd7,3'd7,
        3'd7,3'd2,3'd3,3'd0,3'd1,
        3'd3,3'd0,3'd7,3'd4,3'd4};
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0, start=0;
    reg [7:0] cfg=2;
    reg [W-1:0] idata[0:4];
    reg [4:0] iv=0;
    wire [W-1:0] od[0:4];
    wire [4:0] ov;
    wire running, inj, last, flast;
    wire [31:0] ts, ss;
    fixed_path_switch_5port #(.DATA_WIDTH(W),.NUM_TIMESLOTS(T),
        .SUBSLOT_WIDTH(8),.TAIL_CYCLES(G),.SWITCH_CONFIG(CONFIG)) dut (
        .clk(clk),.rst_n(rst_n),.start(start),.cfg_inject_cycles(cfg),
        .i_north(idata[0]),.i_south(idata[1]),.i_east(idata[2]),
        .i_west(idata[3]),.i_local(idata[4]),
        .i_north_valid(iv[0]),.i_south_valid(iv[1]),.i_east_valid(iv[2]),
        .i_west_valid(iv[3]),.i_local_valid(iv[4]),
        .o_north(od[0]),.o_south(od[1]),.o_east(od[2]),
        .o_west(od[3]),.o_local(od[4]),
        .o_north_valid(ov[0]),.o_south_valid(ov[1]),.o_east_valid(ov[2]),
        .o_west_valid(ov[3]),.o_local_valid(ov[4]),
        .running(running),.inject_enable(inj),.timeslot(ts),.subslot(ss),
        .slot_last(last),.frame_last(flast));
     
    function automatic integer source(input integer slot, output_id);
        case(slot)
            0: case(output_id)
                0,1: source=4;
                3: source=0;
                4: source=3;
                default: source=-1;
            endcase
            1: case(output_id)
                0: source=1;
                1: source=0;
                2: source=3;
                3: source=2;
                default: source=-1;
            endcase
            default: source=-1;
        endcase
    endfunction
    task automatic check_clear;
        for (int j=0;j<5;j++)
            if (ov[j] !== 0 || od[j] !== 0) $fatal(1,"output not cleared: %0d",j);
    endtask
    task automatic run_case(input integer requested);
        integer omega, c, slot, phase, src;
        reg [4:0] ev;
        reg [W-1:0] ed[0:4];
        begin
            omega=(requested==0)?1:requested;
            @(negedge clk); cfg=8'(requested); start=1; iv='1;
            @(posedge clk); #1; check_clear();
            if (!running || ts!==0 || ss!==0) $fatal(1,"start state");
            for(c=0;c<2*T*(omega+G);c++) begin
                @(negedge clk); start=0;
                 
                cfg=8'd19;
                slot=(c/(omega+G))%T; phase=c%(omega+G);
                for(int j=0;j<5;j++) begin
                    idata[j]=32'(c*16+j);
                    iv[j]=((c+j)%4 != 2);
                end
                if(c==0) begin idata[4]=0; iv[4]=1; end
                #1;
                if(ts!==32'(slot) || ss!==32'(phase) || inj!==(phase<omega))
                    $fatal(1,"counter/injection mismatch cycle=%0d",c);
                if(last!==(phase==omega+G-1) || flast!==((slot==T-1)&&(phase==omega+G-1)))
                    $fatal(1,"boundary indication mismatch");
                for(int j=0;j<5;j++) begin
                    src=source(slot,j); ev[j]=0; ed[j]=0;
                    if(src>=0) begin
                        ev[j]=iv[src] && ((src!=4)||(phase<omega));
                        if(ev[j]) ed[j]=idata[src];
                    end
                end
                @(posedge clk); #1;
                for(int j=0;j<5;j++)
                    if(ov[j]!==ev[j] || od[j]!==ed[j])
                        $fatal(1,"switch mismatch cycle=%0d output=%0d",c,j);
            end
            if(ts!==0 || ss!==0) $fatal(1,"non-power-of-two frame wrap failed");
        end
    endtask
    initial begin
        for(int j=0;j<5;j++) idata[j]=0;
        #1; rst_n=1; #1; rst_n=0;
        #1; check_clear();
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        @(posedge clk); #1; check_clear();
        if(running) $fatal(1,"started without start");
        run_case(2); run_case(1); run_case(3); run_case(0);
         
        @(negedge clk); start=1;
        repeat(2) begin @(posedge clk); #1; check_clear(); end
        @(negedge clk); start=0; rst_n=0;
        #1; check_clear();
        if(running || ts!==0 || ss!==0) $fatal(1,"reset failed");
        $display("PASS: switch smoke (multicast, valid, tail, counters, reset/restart)");
        $finish;
    end
    initial begin #100000; $fatal(1,"switch timeout"); end
endmodule
