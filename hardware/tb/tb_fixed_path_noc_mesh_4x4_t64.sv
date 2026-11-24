// =============================================================================
//   High-observability self-checking testbench for the 4x4 TDM mesh (Figure 2).
// =============================================================================
`timescale 1ns/1ps

module tb_fixed_path_noc_mesh_4x4_t64;
    localparam int W = 32, T = 64, G = 7, LIMIT = 2000;

    logic clk = 0;
    always #5 clk = ~clk;

    logic              rst_n = 0, start = 0;
    logic [7:0]        cfg   = 3;
    logic [16*W-1:0]   din   = 0;
    logic [15:0]       vin   = 0;
    wire  [16*W-1:0]   dout;
    wire  [15:0]       vout, ready;
    wire               running, last, flast;
    wire  [31:0]       ts, ss, sdown;

    logic         expected_valid[0:LIMIT-1][0:15];
    logic [W-1:0] expected_data [0:LIMIT-1][0:15];
    int           inj_per_root  [0:15];
    int           rcv_per_dst   [0:15];
    int           accepted, delivered;

    fixed_path_noc_mesh_4x4_t64 #(
        .SUBSLOT_WIDTH(8)
    ) dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .cfg_inject_cycles(cfg),
        .i_local_flat     (din),
        .i_local_valid    (vin),
        .o_local_flat     (dout),
        .o_local_valid    (vout),
        .o_local_ready    (ready),
        .running          (running),
        .timeslot         (ts),
        .subslot          (ss),
        .subslot_down_val (sdown),
        .slot_last        (last),
        .frame_last       (flast)
    );

    function automatic int absval(input int v);
        absval = (v < 0) ? -v : v;
    endfunction

    task automatic run_frame(input int requested);
        int omega, c, slot, phase, root, distance, due;
        logic [15:0]  er;
        logic [W-1:0] token;
        begin
            omega     = (requested == 0) ? 1 : requested;
            accepted  = 0;
            delivered = 0;
            for (int k = 0; k < LIMIT; k++)
                for (int j = 0; j < 16; j++) begin
                    expected_valid[k][j] = 0;
                    expected_data[k][j]  = 0;
                end
            for (int j = 0; j < 16; j++) begin
                inj_per_root[j] = 0;
                rcv_per_dst[j]  = 0;
            end

            $display("\n===============================================================================================");
            $display(" [MESH TEST] requested_omega=%0d (effective omega=%0d, tail=%0d, slot_cycles=%0d, total=%0d)",
                     requested, omega, G, omega + G, 2 * T * (omega + G));
            $display("===============================================================================================");

            @(negedge clk);
            start = 1;
            cfg   = 8'(requested);
            vin   = 0;
            @(posedge clk); #1;
            if (vout !== 0 || dout !== 0 || ts !== 0 || ss !== 0 || sdown !== (omega + G - 1))
                $fatal(1, "restart flush");

            for (c = 0; c < 2 * T * (omega + G); c++) begin
                @(negedge clk);
                start = 0;
                cfg   = 8'd29;
                slot  = (c / (omega + G)) % T;
                phase = c % (omega + G);
                root  = slot % 16;

                vin = '1;
                for (int j = 0; j < 16; j++)
                    din[j*W +: W] = 32'hbad00000 + 32'(j);
                token = 32'(c);
                din[root*W +: W] = token;

                if (omega > 2 && phase == 1)
                    vin[root] = 0;
                er = 0;
                if (phase < omega)
                    er[root] = 1;

                #1;
                if (ready !== er || ts !== 32'(slot) || ss !== 32'(phase) ||
                    sdown !== 32'(omega + G - 1 - phase))
                    $fatal(1, "schedule mismatch cycle=%0d", c);
                if (last  !== (phase == omega + G - 1) ||
                    flast !== ((slot == T - 1) && (phase == omega + G - 1)))
                    $fatal(1, "slot/frame indication mismatch");

                if (er[root] && vin[root]) begin
                    accepted++;
                    inj_per_root[root]++;
                    if (slot < 16 && (c / (T * (omega + G))) == 0) begin
                        $display("  [INJECT] Cycle=%4d | TS=%2d | Subslot=%0d (DownCnt=%0d) | Root=%2d | Token=0x%08x",
                                 c, slot, phase, sdown, root, token);
                    end
                    for (int dst = 0; dst < 16; dst++) begin
                        if (dst != root) begin
                            distance = absval(dst % 4 - root % 4) + absval(dst / 4 - root / 4);
                            due = c + distance;
                            if (due >= LIMIT || expected_valid[due][dst])
                                $fatal(1, "scoreboard collision");
                            expected_valid[due][dst] = 1;
                            expected_data[due][dst]  = token;
                        end
                    end
                end

                @(posedge clk); #1;
                for (int dst = 0; dst < 16; dst++) begin
                    if (vout[dst] !== expected_valid[c][dst])
                        $fatal(1, "valid mismatch c=%0d dst=%0d (got %0b, exp %0b)",
                               c, dst, vout[dst], expected_valid[c][dst]);
                    if (vout[dst]) begin
                        if (dout[dst*W +: W] !== expected_data[c][dst])
                            $fatal(1, "data mismatch c=%0d dst=%0d (got 0x%08x, exp 0x%08x)",
                                   c, dst, dout[dst*W +: W], expected_data[c][dst]);
                        delivered++;
                        rcv_per_dst[dst]++;
                    end else if (dout[dst*W +: W] !== 0) begin
                        $fatal(1, "invalid output not zero");
                    end
                end

                if (flast) begin
                    $display("  [FRAME BOUNDARY] Cycle=%4d (End of TS=%0d) | Cumulative Delivered=%0d/%0d",
                             c, slot, delivered, accepted * 15);
                end
            end

            if (delivered != accepted * 15)
                $fatal(1, "missing/duplicate deliveries");
            if (ts !== 0 || ss !== 0 || vout !== 0)
                $fatal(1, "frame not drained");

            $display("  --- Per-Core Summary (omega=%0d) ---", omega);
            for (int j = 0; j < 16; j++) begin
                $display("    Core %2d: Injected=%3d | Received=%4d", j, inj_per_root[j], rcv_per_dst[j]);
            end
            $display("  [PASS] NoC omega=%0d: %0d injections, %0d checked deliveries", omega, accepted, delivered);
        end
    endtask

    initial begin
        #1; rst_n = 1; #1; rst_n = 0;
        #1; if (vout !== 0 || dout !== 0) $fatal(1, "async reset");
        repeat (2) @(posedge clk);
        @(negedge clk); rst_n = 1;
        @(posedge clk); #1;
        if (running || ready !== 0 || vout !== 0) $fatal(1, "idle");

        run_frame(3);
        run_frame(1);
        run_frame(0);

        @(negedge clk); vin = 1; din = 0; din[31:0] = 32'h1234; start = 0;
        @(posedge clk); #1;
        @(negedge clk); start = 1;
        @(posedge clk); #1;
        if (vout !== 0 || dout !== 0 || ready !== 0)
            $fatal(1, "restart did not flush");
        @(negedge clk); start = 0; vin = 0;
        repeat (10) begin
            @(posedge clk); #1;
            if (vout !== 0) $fatal(1, "stale event after restart");
        end
        @(negedge clk); rst_n = 0;
        #1; if (running || vout !== 0 || ready !== 0) $fatal(1, "reset");
        $display("\n===============================================================================================");
        $display(" PASS: NoC smoke (1-Register Input Buffers, all roots/destinations, tail, frame wrap)");
        $display("===============================================================================================\n");
        $finish;
    end
    initial begin #100000; $fatal(1, "NoC timeout"); end
endmodule