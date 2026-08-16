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
 *
 * The cart icon in the nav is a link to the /cart page (no inline dropdown).
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

    const api = { get, add, remove, clear, contains, updateBadge };
    window.SmartRecoCart = api;
    window.UPulseCart = api;

    document.addEventListener('DOMContentLoaded', updateBadge);
})();