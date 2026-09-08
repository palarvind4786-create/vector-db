/**
 * Vector Database From Scratch - Frontend Dashboard Client
 * Connects UI to FastAPI backend endpoints: /search, /stats, /benchmark/latest
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const nprobeSlider = document.getElementById('nprobe-slider');
    const nprobeVal = document.getElementById('nprobe-val');
    const nprobeGroup = document.getElementById('nprobe-group');
    const topKInput = document.getElementById('top-k-input');
    const toggleBtns = document.querySelectorAll('.toggle-btn');
    const runBenchBtn = document.getElementById('run-bench-btn');

    // Metric Summary Elements
    const metricLatency = document.getElementById('metric-latency');
    const metricScanned = document.getElementById('metric-scanned');
    const metricRecall = document.getElementById('metric-recall');
    const resultsList = document.getElementById('results-list');
    const collectionStatus = document.getElementById('collection-status');
    const clusterChart = document.getElementById('cluster-chart');
    const benchmarkResults = document.getElementById('benchmark-results');

    let activeIndexMode = 'ivf';
    let dbStats = null;

    // --- 1. Initial State & Statistics ---
    async function loadStats() {
        try {
            const resp = await fetch('/stats');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            dbStats = await resp.json();

            if (collectionStatus) {
                collectionStatus.textContent = `Engine: Ready (${dbStats.active_vectors.toLocaleString()} vectors | ${dbStats.num_clusters} clusters)`;
            }

            renderClusterDistribution(dbStats);
        } catch (err) {
            console.warn('Failed to load DB stats:', err);
            if (collectionStatus) {
                collectionStatus.textContent = 'Engine: Offline or Initializing...';
            }
        }
    }

    function renderClusterDistribution(stats) {
        if (!clusterChart || !stats.cluster_distribution) return;

        const sizes = Object.values(stats.cluster_distribution);
        const maxVal = Math.max(...sizes, 1);

        let html = '<div class="cluster-bar-grid">';
        sizes.forEach((size, idx) => {
            const pct = Math.max(4, Math.round((size / maxVal) * 100));
            html += `<div class="cluster-bar" style="height: ${pct}%;" title="Cluster #${idx}: ${size} vectors"></div>`;
        });
        html += '</div>';
        html += `<p class="section-desc" style="margin-top: 0.5rem; text-align: center;">Distribution of ${stats.active_vectors.toLocaleString()} vectors across ${sizes.length} Voronoi clusters (mean: ${stats.mean_cluster_size.toFixed(1)} vecs/cluster, empty: ${stats.empty_clusters})</p>`;
        clusterChart.innerHTML = html;
    }

    loadStats();

    // --- 2. Controls & Sliders ---
    if (nprobeSlider && nprobeVal) {
        nprobeSlider.addEventListener('input', (e) => {
            nprobeVal.textContent = e.target.value;
        });
    }

    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            toggleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeIndexMode = btn.dataset.index;

            // Hide nprobe slider for exact search
            if (nprobeGroup) {
                nprobeGroup.style.display = activeIndexMode === 'exact' ? 'none' : 'flex';
            }
        });
    });

    // --- 3. Semantic Search Execution ---
    async function executeSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            searchInput.focus();
            return;
        }

        const k = parseInt(topKInput.value) || 10;
        const nprobe = parseInt(nprobeSlider.value) || 8;

        // UI Loading State
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<span class="loading-spinner"></span>Searching...';
        resultsList.innerHTML = '<p class="placeholder-text">Searching indexed embeddings...</p>';

        try {
            if (activeIndexMode === 'exact') {
                // Exact brute-force search
                const resp = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, k, mode: 'exact' }),
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || `Server returned ${resp.status}`);
                }

                const data = await resp.json();
                renderMetrics({
                    latency: `${data.latency_ms.toFixed(2)} ms`,
                    scanned: `${data.candidates_examined.toLocaleString()} (100%)`,
                    recall: '100.0%',
                });
                renderResultsList(data.results, 'Exact');

            } else if (activeIndexMode === 'ivf') {
                // IVF search + Exact search in parallel to evaluate Recall vs Exact
                const [ivfResp, exactResp] = await Promise.all([
                    fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, k, mode: 'ivf', nprobe }),
                    }),
                    fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, k, mode: 'exact' }),
                    }),
                ]);

                if (!ivfResp.ok) {
                    const err = await ivfResp.json();
                    throw new Error(err.detail || `Server returned ${ivfResp.status}`);
                }

                const ivfData = await ivfResp.json();
                let recallPct = '--%';
                let totalExamined = ivfData.candidates_examined;
                let totalDataset = 10000;

                if (exactResp.ok) {
                    const exactData = await exactResp.json();
                    totalDataset = exactData.candidates_examined || totalDataset;
                    const exactIdSet = new Set(exactData.results.map(r => r.id));
                    const matches = ivfData.results.filter(r => exactIdSet.has(r.id)).length;
                    recallPct = `${((matches / Math.max(1, exactData.results.length)) * 100).toFixed(1)}%`;
                }

                const pctExamined = ((totalExamined / totalDataset) * 100).toFixed(1);
                renderMetrics({
                    latency: `${ivfData.latency_ms.toFixed(2)} ms`,
                    scanned: `${totalExamined.toLocaleString()} (${pctExamined}%)`,
                    recall: recallPct,
                });
                renderResultsList(ivfData.results, `IVF-Flat (nprobe=${nprobe})`);

            } else if (activeIndexMode === 'compare') {
                // Side-by-side comparison mode
                const [ivfResp, exactResp] = await Promise.all([
                    fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, k, mode: 'ivf', nprobe }),
                    }),
                    fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query, k, mode: 'exact' }),
                    }),
                ]);

                if (!ivfResp.ok || !exactResp.ok) {
                    throw new Error('Comparison queries failed.');
                }

                const ivfData = await ivfResp.json();
                const exactData = await exactResp.json();

                const exactIdSet = new Set(exactData.results.map(r => r.id));
                const matches = ivfData.results.filter(r => exactIdSet.has(r.id)).length;
                const recallPct = `${((matches / Math.max(1, exactData.results.length)) * 100).toFixed(1)}%`;
                const pctExamined = ((ivfData.candidates_examined / exactData.candidates_examined) * 100).toFixed(1);

                renderMetrics({
                    latency: `${ivfData.latency_ms.toFixed(2)}ms vs ${exactData.latency_ms.toFixed(2)}ms`,
                    scanned: `${ivfData.candidates_examined.toLocaleString()} vs ${exactData.candidates_examined.toLocaleString()}`,
                    recall: recallPct,
                });
                renderCompareView(ivfData.results, exactData.results, exactIdSet, nprobe);
            }
        } catch (err) {
            console.error('Search failed:', err);
            resultsList.innerHTML = `<div class="error-msg"><strong>Search Error:</strong> ${err.message}</div>`;
        } finally {
            searchBtn.disabled = false;
            searchBtn.textContent = 'Search';
        }
    }

    function renderMetrics({ latency, scanned, recall }) {
        if (metricLatency) metricLatency.textContent = latency;
        if (metricScanned) metricScanned.textContent = scanned;
        if (metricRecall) metricRecall.textContent = recall;
    }

    function renderResultsList(results, modeLabel) {
        if (!results || results.length === 0) {
            resultsList.innerHTML = '<p class="placeholder-text">No matching documents found.</p>';
            return;
        }

        let html = '';
        results.forEach(item => {
            const category = item.metadata ? item.metadata.category : null;
            const categoryTag = category ? `<span class="result-badge">${escapeHtml(category)}</span>` : '';
            const textContent = item.text || (item.metadata ? item.metadata.text : null) || 'No text snippet available.';

            html += `
                <div class="result-item">
                    <div class="result-meta">
                        <span class="result-rank">#${item.rank}</span>
                        <span class="result-score">Score: ${item.score.toFixed(4)}</span>
                        ${categoryTag}
                        <span class="result-id">${escapeHtml(item.id)}</span>
                    </div>
                    <p class="result-text">${escapeHtml(textContent)}</p>
                </div>
            `;
        });
        resultsList.innerHTML = html;
    }

    function renderCompareView(ivfResults, exactResults, exactIdSet, nprobe) {
        let html = '<div class="compare-container">';

        // IVF Column
        html += `<div class="compare-column"><h3>IVF-Flat (nprobe=${nprobe}) <span>ANN Approx</span></h3><div class="results-list">`;
        ivfResults.forEach(item => {
            const isMatch = exactIdSet.has(item.id);
            const matchBadge = isMatch ? '<span class="result-badge" style="color: var(--accent-green); border-color: rgba(16,185,129,0.4)">In Ground Truth</span>' : '';
            html += `
                <div class="result-item">
                    <div class="result-meta">
                        <span class="result-rank">#${item.rank}</span>
                        <span class="result-score">${item.score.toFixed(4)}</span>
                        ${matchBadge}
                        <span class="result-id">${escapeHtml(item.id)}</span>
                    </div>
                    <p class="result-text">${escapeHtml(item.text || '')}</p>
                </div>
            `;
        });
        html += '</div></div>';

        // Exact Column
        html += '<div class="compare-column"><h3>Exact (Brute-force) <span>Ground Truth</span></h3><div class="results-list">';
        exactResults.forEach(item => {
            html += `
                <div class="result-item">
                    <div class="result-meta">
                        <span class="result-rank">#${item.rank}</span>
                        <span class="result-score">${item.score.toFixed(4)}</span>
                        <span class="result-id">${escapeHtml(item.id)}</span>
                    </div>
                    <p class="result-text">${escapeHtml(item.text || '')}</p>
                </div>
            `;
        });
        html += '</div></div></div>';

        resultsList.innerHTML = html;
    }

    // Helper: Escape HTML
    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Event Listeners for Search
    if (searchBtn) {
        searchBtn.addEventListener('click', executeSearch);
    }

    if (searchInput) {
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                executeSearch();
            }
        });
    }

    // --- 4. Benchmark Execution ---
    if (runBenchBtn) {
        runBenchBtn.addEventListener('click', async () => {
            runBenchBtn.disabled = true;
            runBenchBtn.innerHTML = '<span class="loading-spinner"></span>Loading Benchmark...';
            benchmarkResults.innerHTML = '<p class="placeholder-text">Retrieving 500-query benchmark results...</p>';

            try {
                const resp = await fetch('/benchmark/latest');
                if (!resp.ok) throw new Error(`Benchmark data not found (${resp.status})`);
                const data = await resp.json();

                renderBenchmarkTable(data);
            } catch (err) {
                console.error('Failed to load benchmark:', err);
                benchmarkResults.innerHTML = `<div class="error-msg"><strong>Error:</strong> Could not load benchmark records. Run the benchmark suite in terminal first.</div>`;
            } finally {
                runBenchBtn.disabled = false;
                runBenchBtn.textContent = 'Run Benchmark Suite';
            }
        });
    }

    function renderBenchmarkTable(data) {
        if (!benchmarkResults || !data.records) return;

        let html = `
            <table class="bench-table">
                <thead>
                    <tr>
                        <th>Configuration</th>
                        <th>nprobe</th>
                        <th>Recall@10</th>
                        <th>Mean Latency</th>
                        <th>P50</th>
                        <th>P95</th>
                        <th>Examined</th>
                        <th>Speedup</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.records.forEach(r => {
            const nprobeStr = r.nprobe !== null && r.nprobe !== undefined ? r.nprobe : '-';
            const recallStr = (r.recall_at_10 * 100).toFixed(1) + '%';
            const meanStr = r.avg_latency_ms.toFixed(3) + ' ms';
            const p50Str = r.p50_latency_ms.toFixed(3) + ' ms';
            const p95Str = r.p95_latency_ms.toFixed(3) + ' ms';
            const speedupStr = r.speedup.toFixed(2) + 'x';
            const examinedStr = `${r.avg_candidates.toFixed(0)} (${r.pct_examined.toFixed(1)}%)`;

            html += `
                <tr>
                    <td style="color: var(--text-primary); font-weight: 500;">${escapeHtml(r.name)}</td>
                    <td>${nprobeStr}</td>
                    <td style="color: var(--accent-green); font-weight: 600;">${recallStr}</td>
                    <td>${meanStr}</td>
                    <td>${p50Str}</td>
                    <td>${p95Str}</td>
                    <td>${examinedStr}</td>
                    <td style="color: var(--accent-blue); font-weight: 600;">${speedupStr}</td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        benchmarkResults.innerHTML = html;
    }
});
