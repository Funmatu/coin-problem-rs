import init, { solve_js } from './pkg/limit_break_rs.js';

async function run() {
    await init();
    
    const btn = document.getElementById('run-btn');
    const output = document.getElementById('output');
    
    btn.innerText = "RUN SOLVER";
    btn.disabled = false;

    btn.addEventListener('click', () => {
        const target = parseInt(document.getElementById('input-target').value);
        const maxCoins = parseInt(document.getElementById('input-max-coins').value);
        const coinsStr = document.getElementById('input-coins').value;
        
        // Parse coins string to BigInt64Array (Rust expects i64)
        const coinsArray = coinsStr.split(',').map(s => BigInt(s.trim()));
        const coins = new BigInt64Array(coinsArray);

        output.innerText = "Computing...";
        
        setTimeout(() => {
            const start = performance.now();
            try {
                // Call Rust
                const result = solve_js(target, maxCoins, coins);
                const end = performance.now();
                output.innerText = `Combinations: ${result}\nTime: ${(end - start).toFixed(2)} ms`;
            } catch (e) {
                output.innerText = `Error: ${e}`;
            }
        }, 50);
    });
}

run();