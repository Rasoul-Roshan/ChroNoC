// Self-checking smoke test for the mesh.
`timescale 1ns/1ps
module tb_fixed_path_noc_mesh_4x4_t64;
    localparam W=32, T=64, G=7, LIMIT=2000;
    reg clk=0; always #5 clk=~clk;
    reg rst_n=0, start=0;
    reg [7:0] cfg=3;
    reg [16*W-1:0] din=0;
    reg [15:0] vin=0;
    wire [16*W-1:0] dout;
    wire [15:0] vout, ready;
    wire running, last, flast;
    wire [31:0] ts, ss;
    reg expected_valid[0:LIMIT-1][0:15];
    reg [W-1:0] expected_data[0:LIMIT-1][0:15];
    integer accepted, delivered;
    fixed_path_noc_mesh_4x4_t64 #(.SUBSLOT_WIDTH(8)) dut (
        .clk(clk),.rst_n(rst_n),.start(start),.cfg_inject_cycles(cfg),
        .i_local_flat(din),.i_local_valid(vin),.o_local_flat(dout),
        .o_local_valid(vout),.o_local_ready(ready),.running(running),
        .timeslot(ts),.subslot(ss),.slot_last(last),.frame_last(flast));
    function automatic integer absval(input integer v);
        absval=(v<0)?-v:v;
    endfunction
    task automatic run_frame(input integer requested);
        integer omega, c, slot, phase, root, distance, due;
        reg [15:0] er;
        reg [W-1:0] token;
        begin
            omega=(requested==0)?1:requested;
            accepted=0; delivered=0;
            for(int k=0;k<LIMIT;k++) for(int j=0;j<16;j++) begin
                expected_valid[k][j]=0; expected_data[k][j]=0;
            end
            @(negedge clk); start=1; cfg=8'(requested); vin=0;
            @(posedge clk); #1;
            if(vout!==0 || dout!==0 || ts!==0 || ss!==0) $fatal(1,"restart flush");
             
            for(c=0;c<2*T*(omega+G);c++) begin
                @(negedge clk); start=0; cfg=8'd29;
                slot=(c/(omega+G))%T; phase=c%(omega+G); root=slot%16;
                 
                vin='1;
                for(int j=0;j<16;j++) din[j*W +: W]=32'hbad00000+32'(j);
                token=32'(c); din[root*W +: W]=token;
                 
                if(omega>2 && phase==1) vin[root]=0;
                er=0; if(phase<omega) er[root]=1;
                #1;
                if(ready!==er || ts!==32'(slot) || ss!==32'(phase))
                    $fatal(1,"schedule mismatch cycle=%0d",c);
                if(last!==(phase==omega+G-1) || flast!==((slot==T-1)&&(phase==omega+G-1)))
                    $fatal(1,"slot/frame indication mismatch");
                if(er[root] && vin[root]) begin
                    accepted++;
                    for(int dst=0;dst<16;dst++) if(dst!=root) begin
                        distance=absval(dst%4-root%4)+absval(dst/4-root/4);

                        due=c+distance;
                        if(due>=LIMIT || expected_valid[due][dst]) $fatal(1,"scoreboard collision");
                        expected_valid[due][dst]=1;
                        expected_data[due][dst]=token;
                    end
                end
                @(posedge clk); #1;
                for(int dst=0;dst<16;dst++) begin
                    if(vout[dst]!==expected_valid[c][dst])
                        $fatal(1,"valid mismatch c=%0d dst=%0d",c,dst);
                    if(vout[dst]) begin
                        if(dout[dst*W +: W]!==expected_data[c][dst])
                            $fatal(1,"data mismatch c=%0d dst=%0d",c,dst);
                        delivered++;
                    end else if(dout[dst*W +: W]!==0) $fatal(1,"invalid output not zero");
                end
            end
            if(delivered!=accepted*15) $fatal(1,"missing/duplicate deliveries");
            if(ts!==0 || ss!==0 || vout!==0) $fatal(1,"frame not drained");
            $display("NoC omega=%0d: %0d injections, %0d checked deliveries",omega,accepted,delivered);
        end
    endtask
    initial begin
        #1; rst_n=1; #1; rst_n=0;
        #1; if(vout!==0 || dout!==0) $fatal(1,"async reset");
        repeat(2) @(posedge clk);
        @(negedge clk); rst_n=1;
        @(posedge clk); #1;
        if(running || ready!==0 || vout!==0) $fatal(1,"idle");
        run_frame(3); run_frame(1); run_frame(0);
         
        @(negedge clk); vin=1; din=0; din[31:0]=32'h1234; start=0;
        @(posedge clk); #1;
        @(negedge clk); start=1;
        @(posedge clk); #1;
        if(vout!==0 || dout!==0 || ready!==0) $fatal(1,"restart did not flush");
        @(negedge clk); start=0; vin=0;
        repeat(10) begin @(posedge clk); #1; if(vout!==0) $fatal(1,"stale event after restart"); end
        @(negedge clk); rst_n=0;
        #1; if(running || vout!==0 || ready!==0) $fatal(1,"reset");
        $display("PASS: NoC smoke (all roots/destinations, last subslot, tail, frame wrap)");
        $finish;
    end
    initial begin #100000; $fatal(1,"NoC timeout"); end
endmodule
