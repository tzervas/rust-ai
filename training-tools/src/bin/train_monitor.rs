//! Training monitor TUI binary.
//!
//! Usage:
//!   train-monitor [--runs-dir <path>]
//!
//! Hotkeys:
//!   q - Quit
//!   j/k or arrows - Navigate runs
//!   r - Refresh
//!   c - Clear history

use std::path::PathBuf;

use clap::Parser;

use training_tools::live_monitor::LiveMonitor;

#[derive(Parser)]
#[command(name = "train-monitor")]
#[command(about = "Live streaming TUI for monitoring rust-ai training runs")]
struct Args {
    /// Directory containing training runs
    #[arg(short, long, default_value = "./runs")]
    runs_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Ensure runs directory exists
    std::fs::create_dir_all(&args.runs_dir)?;

    // Create and run live monitor
    let mut monitor = LiveMonitor::new(args.runs_dir);
    monitor.run()?;

    Ok(())
}
