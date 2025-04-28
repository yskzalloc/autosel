// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

// Setup logging with optional debug level
pub fn setup_logging(debug: bool, batch_mode: bool) {
    let mut builder = env_logger::Builder::from_default_env();

    if debug {
        builder.filter_level(log::LevelFilter::Debug);
    } else if batch_mode {
        // In batch mode, only show warnings and errors
        builder.filter_level(log::LevelFilter::Warn);
    } else {
        builder.filter_level(log::LevelFilter::Info);
    }

    builder.init();
}

#[cfg(test)]
mod tests {
    // Test is limited because env_logger can only be initialized once
    #[test]
    fn test_setup_logging() {
        // This test just ensures the code compiles and doesn't panic
        // We can't call setup_logging directly in tests because env_logger can only be initialized once

        // Instead we'll just verify that the function exists and has the right signature
        let _setup_fn: fn(bool, bool) = super::setup_logging;

        // In a real test environment, we might use a mocking framework to verify behavior
    }
}