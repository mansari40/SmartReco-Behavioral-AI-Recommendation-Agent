/**
 * Behavioral event tracker.
 *
 * Tracks:
 * - page views
 * - product/course views
 * - course-card clicks
 * - searches
 * - add-to-cart intent
 * - checkout starts
 * - time spent
 *
 * Events are batched client-side and sent to POST /api/events.
 *
 * The tracker also exposes window.UPulseTracker so page-specific
 * scripts can record events and render the shared "Your Signal" widget.
 * window.SmartRecoTracker is kept as an alias for backward compatibility
 * with any pages not yet migrated off the old name.
 */

(function () {
    const FLUSH_INTERVAL_MS = 5000;
    const API_URL = '/api/events';

    let queue = [];
    let pageEnteredAt = Date.now();

    // ------------------------------------------------------------
    // AUTH
    // ------------------------------------------------------------

    function getToken() {
        return localStorage.getItem('access_token');
    }

    // ------------------------------------------------------------
    // HUMAN-READABLE PAGE LABEL
    // ------------------------------------------------------------

    function getPageLabel() {
        const path = window.location.pathname;

        if (path === '/') {
            return 'Catalog';
        }

        if (path === '/recommendations') {
            return 'For You';
        }

        if (path === '/login') {
            return 'Login';
        }

        if (path === '/register') {
            return 'Create Account';
        }

        if (path.startsWith('/products/')) {
            return (
                window.__upulseProduct?.title ||
                window.__smartrecoProduct?.title ||
                'Course'
            );
        }

        return path;
    }

    // ------------------------------------------------------------
    // EVENT QUEUE
    // ------------------------------------------------------------

    function queueEvent(
        eventType,
        productId,
        metadata
    ) {
        queue.push({
            event_type: eventType,
            product_id: productId || null,
            event_metadata: metadata || {},
        });

        try {
            if (
                window.UPulseTracker &&
                typeof window.UPulseTracker.onEvent === 'function'
            ) {
                window.UPulseTracker.onEvent();
            }
        } catch (err) {
            console.error(
                'UPulseTracker callback error:',
                err
            );
        }
    }

    // ------------------------------------------------------------
    // FLUSH EVENTS
    // ------------------------------------------------------------

    async function flush(useKeepalive = false) {
        if (queue.length === 0) {
            return;
        }

        const token = getToken();

        if (!token) {
            queue = [];
            return;
        }

        const events = queue;
        queue = [];

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + token,
                },
                body: JSON.stringify({
                    events: events,
                }),
                keepalive: useKeepalive,
            });

            if (!response.ok) {
                console.error(
                    'Event ingestion failed:',
                    response.status
                );

                queue = events.concat(queue);
            }
        } catch (error) {
            console.error(
                'Event ingestion error:',
                error
            );

            queue = events.concat(queue);
        }
    }

    // ------------------------------------------------------------
    // PAGE VIEW
    // ------------------------------------------------------------

    queueEvent(
        'page_view',
        null,
        {
            path: window.location.pathname,
            page_name: getPageLabel(),
        }
    );

    // ------------------------------------------------------------
    // PRODUCT VIEW
    // ------------------------------------------------------------

    const currentProduct = window.__upulseProduct || window.__smartrecoProduct;

    if (currentProduct && currentProduct.id) {
        queueEvent(
            'product_view',
            currentProduct.id,
            {
                title: currentProduct.title || 'Course',
                category: currentProduct.category || '',
            }
        );
    }

    // ------------------------------------------------------------
    // COURSE / PRODUCT CARD CLICK
    // ------------------------------------------------------------

    document.addEventListener(
        'click',
        function (event) {
            const card =
                event.target.closest(
                    '.product-card, .course-card'
                );

            if (
                !card ||
                !card.dataset.productId
            ) {
                return;
            }

            const title =
                card.dataset.title ||
                card.querySelector('h3')?.textContent?.trim() ||
                'Course';

            const category =
                card.dataset.category ||
                '';

            queueEvent(
                'click',
                card.dataset.productId,
                {
                    title: title,
                    category: category,
                    path: window.location.pathname,
                }
            );
        }
    );

    // ------------------------------------------------------------
    // TIME SPENT
    // ------------------------------------------------------------

    window.addEventListener(
        'beforeunload',
        function () {
            const seconds = Math.round(
                (Date.now() - pageEnteredAt) / 1000
            );

            queueEvent(
                'time_spent',
                null,
                {
                    seconds: seconds,
                    path: window.location.pathname,
                    page_name: getPageLabel(),
                }
            );

            flush(true);
        }
    );

    // ------------------------------------------------------------
    // BACKGROUND FLUSH
    // ------------------------------------------------------------

    setInterval(
        function () {
            flush(false);
        },
        FLUSH_INTERVAL_MS
    );

    // ------------------------------------------------------------
    // SEARCH
    // ------------------------------------------------------------

    function trackSearch(
        query,
        category
    ) {
        queueEvent(
            'search',
            null,
            {
                query: query,
                category:
                    category || null,
            }
        );
    }

    // ------------------------------------------------------------
    // ADD TO CART
    // ------------------------------------------------------------

    function trackAddToCart(
        productId,
        title,
    ) {
        queueEvent(
            'add_to_cart',
            productId,
            {
                title:
                    title || 'Course',
            }
        );
        // Add-to-cart is a high-intent signal, so send it immediately.
        flush(false);
    }

    // ------------------------------------------------------------
    // CHECKOUT / PURCHASE INTENT
    // ------------------------------------------------------------

    function trackCheckout(
        productId,
        title
    ) {
        queueEvent(
            'checkout_start',
            productId,
            {
                title:
                    title || 'Course',
            }
        );

        // Checkout intent is high-intent, so send it immediately.
        flush(false);
    }

    // ------------------------------------------------------------
    // HTML ESCAPING
    // ------------------------------------------------------------

    function escapeHtml(str) {
        const div =
            document.createElement('div');

        div.textContent =
            String(str ?? '');

        return div.innerHTML;
    }

    // ------------------------------------------------------------
    // HUMAN-READABLE EVENT DESCRIPTION
    // ------------------------------------------------------------

    function describeEvent(event) {
        const meta =
            event.event_metadata || {};

        switch (event.event_type) {

            case 'page_view':
                return {
                    label: 'Page',
                    detail:
                        meta.page_name ||
                        'Catalog',
                };

            case 'product_view':
                return {
                    label: 'Viewed',
                    detail:
                        meta.title ||
                        meta.category ||
                        'Course',
                };

            case 'click':
                return {
                    label: 'Clicked',
                    detail:
                        meta.title ||
                        'Course',
                };

            case 'search':
                return {
                    label: 'Searched',
                    detail:
                        meta.query ||
                        '',
                };

            case 'add_to_cart':
                return {
                    label: 'Cart',
                    detail:
                        meta.title ||
                        'Course',
                };

            case 'checkout_start':
                return {
                    label: 'Checkout',
                    detail:
                        meta.title ||
                        'Course',
                };

            case 'time_spent':
                return {
                    label: 'Time',
                    detail:
                        `${meta.seconds || 0}s on ${
                            meta.page_name ||
                            meta.title ||
                            'page'
                        }`,
                };

            default:
                return {
                    label: 'Activity',
                    detail:
                        meta.page_name ||
                        meta.title ||
                        'UPulse',
                };
        }
    }

    // ------------------------------------------------------------
    // SIGNAL ROW
    // ------------------------------------------------------------

    function signalRowHtml(
        event,
        opts
    ) {
        const {
            label,
            detail
        } =
            describeEvent(event);

        const time =
            event.created_at
                ? new Date(
                      event.created_at
                  ).toLocaleTimeString(
                      [],
                      {
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                      }
                  )
                : '';

        const highlight =
            opts &&
            opts.highlightProductId &&
            event.product_id ===
                opts.highlightProductId
                ? ' signal-highlight'
                : '';

        return `
            <div class="signal-row${highlight}">
                <span class="signal-label">
                    ${escapeHtml(label)}
                </span>

                <span class="signal-detail">
                    ${escapeHtml(detail)}
                </span>

                <span class="signal-time">
                    ${escapeHtml(time)}
                </span>
            </div>
        `;
    }

    // ------------------------------------------------------------
    // SIGNAL WIDGET
    // ------------------------------------------------------------

    function renderSignal(
        containerEl,
        events,
        opts
    ) {
        if (!containerEl) {
            return;
        }

        opts = opts || {};

        const title =
            opts.title ||
            'Your Signal';

        const limit =
            opts.limit || 8;

        const items =
            (events || []).slice(
                0,
                limit
            );

        let html = `
            <div class="signal-header">
                <span class="signal-title">
                    ${escapeHtml(title)}
                </span>

                <span class="signal-live">
                    <span class="pulse-dot-sm"></span>
                    streaming
                </span>
            </div>

            <div class="signal-rows">
        `;

        if (items.length === 0) {
            html += `
                <div class="signal-empty">
                    No signal yet — browse a bit and
                    this fills in live.
                </div>
            `;
        } else {
            html += items
                .map(function (event) {
                    return signalRowHtml(
                        event,
                        opts
                    );
                })
                .join('');
        }

        html += '</div>';

        containerEl.innerHTML = html;
    }

    // ------------------------------------------------------------
    // PUBLIC API
    // ------------------------------------------------------------

    const api = {
        trackSearch,
        trackAddToCart,
        trackCheckout,
        renderSignal,
        onEvent: null,
    };

    window.UPulseTracker = api;
    window.SmartRecoTracker = api; // backward-compat alias

})();