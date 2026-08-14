/**
 * Client-side cart. There is no server-side cart model in this app — the
 * cart is just a list of product IDs in localStorage; full product details
 * (title/price/category) are fetched live from /api/products whenever
 * something needs to be displayed. This keeps the cart from ever showing
 * stale price/title data if a product is edited later.
 *
 * Primary API is window.SmartRecoCart (that's what cart.html and
 * product_detail.html already call). window.UPulseCart is an alias.
 * Changes fire both 'smartreco-cart' and 'upulse-cart' window events so any
 * page can react — cart.html listens for these to refresh its list.
 */
(function () {
    const CART_KEY = 'upulse_cart_ids';

    function get() {
        try {
            const raw = localStorage.getItem(CART_KEY);
            const parsed = raw ? JSON.parse(raw) : [];
            return Array.isArray(parsed) ? parsed : [];
        } catch (err) {
            return [];
        }
    }

    function _save(ids) {
        localStorage.setItem(CART_KEY, JSON.stringify(ids));
        updateBadge();
        renderDropdown();
        window.dispatchEvent(new Event('smartreco-cart'));
        window.dispatchEvent(new Event('upulse-cart'));
    }

    function add(id) {
        if (!id) return;
        const ids = get();
        if (!ids.includes(id)) {
            ids.push(id);
            _save(ids);
        }
    }

    function remove(id) {
        _save(get().filter((existing) => existing !== id));
    }

    function clear() {
        _save([]);
    }

    function contains(id) {
        return get().includes(id);
    }

    function updateBadge() {
        const badge = document.getElementById('cart-count');
        if (!badge) return;
        const count = get().length;
        badge.textContent = String(count);
        badge.style.display = count > 0 ? 'inline-flex' : 'none';
    }

    // ------------------------------------------------------------
    // NAV DROPDOWN — fetches product details so items are recognizable,
    // not just raw IDs.
    // ------------------------------------------------------------

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = String(str ?? '');
        return div.innerHTML;
    }

    let productCache = null;
    let productCacheAt = 0;
    const CACHE_TTL_MS = 30000; // avoid refetching the whole catalog on every dropdown open

    async function fetchProducts() {
        const now = Date.now();
        if (productCache && now - productCacheAt < CACHE_TTL_MS) {
            return productCache;
        }
        try {
            const res = await fetch('/api/products');
            productCache = res.ok ? await res.json() : [];
            productCacheAt = now;
        } catch (err) {
            productCache = productCache || [];
        }
        return productCache;
    }

    async function renderDropdown() {
        const container = document.getElementById('cart-dropdown-items');
        if (!container) return; // nav dropdown isn't on every page state

        const ids = get();
        if (ids.length === 0) {
            container.innerHTML = '<div class="cart-dropdown-empty">Your cart is empty.</div>';
            return;
        }

        container.innerHTML = '<div class="cart-dropdown-empty">Loading...</div>';

        const all = await fetchProducts();
        const byId = {};
        all.forEach((p) => { byId[p.id] = p; });
        const items = ids.map((id) => byId[id]).filter(Boolean);

        // Stored IDs that no longer exist in the catalog (deleted product)
        // are silently dropped from display but left in storage — a
        // background sync could prune them properly; for now this at
        // least avoids showing broken rows.
        if (items.length === 0) {
            container.innerHTML = '<div class="cart-dropdown-empty">Your cart is empty.</div>';
            return;
        }

        container.innerHTML = items.map((item) => (
            '<div class="cart-item" data-product-id="' + escapeHtml(item.id) + '">' +
                '<div class="cart-item-info">' +
                    '<div class="cart-item-title">' + escapeHtml(item.title) + '</div>' +
                    '<div class="cart-item-meta">' + escapeHtml(item.category || '') + ' &middot; $' + Number(item.price || 0).toFixed(2) + '</div>' +
                '</div>' +
                '<button type="button" class="cart-item-remove" aria-label="Remove ' + escapeHtml(item.title) + '" data-remove="' + escapeHtml(item.id) + '">' +
                    '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>' +
                '</button>' +
            '</div>'
        )).join('');

        container.querySelectorAll('[data-remove]').forEach((btn) => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                remove(btn.getAttribute('data-remove'));
            });
        });
    }

    const api = { get, add, remove, clear, contains, updateBadge, renderDropdown };
    window.SmartRecoCart = api;
    window.UPulseCart = api;

    document.addEventListener('DOMContentLoaded', updateBadge);
})();