/**
 * Behavioral event tracker. Watches page views, product-card clicks, and
 * time spent, batches everything client-side, and flushes to
 * POST /api/events on an interval or on page unload — never one network
 * call per event. This is what makes tracking non-blocking: nothing here
 * waits on the network before the user can keep interacting with the page.
 */
(function () {
    const FLUSH_INTERVAL_MS = 5000;
    const API_URL = '/api/events';

    let queue = [];
    let pageEnteredAt = Date.now();

    function getToken() {
        return localStorage.getItem('access_token');
    }

    function queueEvent(eventType, productId, metadata) {
        queue.push({
            event_type: eventType,
            product_id: productId || null,
            event_metadata: metadata || {},
        });
    }

    function flush(useBeacon) {
        if (queue.length === 0) return;
        const token = getToken();
        if (!token) {
            // Not logged in — drop the queue rather than sending
            // unauthenticated events that will just 401.
            queue = [];
            return;
        }

        const payload = JSON.stringify({ events: queue });
        queue = [];

        if (useBeacon && navigator.sendBeacon) {
            // sendBeacon can't set custom headers (no Authorization), so it
            // only works for the final unload flush if we accept losing
            // auth on that specific call. We prefer fetch with keepalive
            // instead, which does support headers.
            fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + token,
                },
                body: payload,
                keepalive: true,
            }).catch(() => {});
            return;
        }

        fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token,
            },
            body: payload,
        }).catch(() => {
            // Silently drop on failure — tracking must never surface
            // errors to the user or break the page.
        });
    }

    // --- Track page view on load ---
    queueEvent('page_view', null, { path: window.location.pathname });

    // --- Track clicks on product cards ---
    document.addEventListener('click', function (e) {
        const card = e.target.closest('.product-card');
        if (card) {
            queueEvent('click', card.dataset.productId, { path: window.location.pathname });
        }
    });

    // --- Track time spent on page, sent on unload ---
    window.addEventListener('beforeunload', function () {
        const timeSpentSeconds = Math.round((Date.now() - pageEnteredAt) / 1000);
        queueEvent('time_spent', null, { seconds: timeSpentSeconds, path: window.location.pathname });
        flush(true);
    });

    // --- Periodic flush, so events aren't only sent on unload ---
    setInterval(function () {
        flush(false);
    }, FLUSH_INTERVAL_MS);
})();