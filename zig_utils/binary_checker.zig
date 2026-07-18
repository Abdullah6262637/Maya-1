const std = @import("std");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    
    // Attempt to open the binary token file
    const bin_path = "../shared/mock_data.bin";
    var file = std.fs.cwd().openFile(bin_path, .{}) catch |err| {
        try stdout.print("Error: Failed to open file at '{s}': {}\n", .{ bin_path, err });
        std.process.exit(1);
    };
    defer file.close();

    const size = try file.getEndPos();
    if (size % 4 != 0) {
        try stdout.print("Verification Failure: File size ({}) is not a multiple of 4 bytes!\n", .{size});
        std.process.exit(1);
    }

    const num_tokens = size / 4;
    try stdout.print("Validation Successful!\n", .{});
    try stdout.print("  Binary File Size:  {} bytes\n", .{size});
    try stdout.print("  Token Count:        {} tokens\n", .{num_tokens});

    // Heap allocation for sample inspection
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const inspect_count = if (num_tokens < 10) num_tokens else 10;
    const buffer = try allocator.alloc(u32, inspect_count);
    defer allocator.free(buffer);

    // Cast buffer slice to raw byte slice for low-level reading
    const byte_slice = std.mem.sliceAsBytes(buffer);
    const bytes_read = try file.read(byte_slice);

    if (bytes_read != inspect_count * 4) {
        try stdout.print("Error: Read count mismatch (read {}, expected {})\n", .{ bytes_read, inspect_count * 4 });
        std.process.exit(1);
    }

    try stdout.print("  First 10 tokens:   ", .{});
    for (buffer) |token| {
        try stdout.print("{} ", .{token});
    }
    try stdout.print("\n", .{});

    // Sanity checks on vocab bounds
    var has_anomaly = false;
    for (buffer) |token| {
        if (token > 32000) {
            try stdout.print("  [ANOMALY] Token ID {} exceeds standard LLaMA vocab range (32000).\n", .{token});
            has_anomaly = true;
        }
    }

    if (!has_anomaly) {
        try stdout.print("  Check completed: All checked tokens are within valid vocab bounds.\n", .{});
    }
}
