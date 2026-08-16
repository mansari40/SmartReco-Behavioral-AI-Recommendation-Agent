/**
 * Client-side wishlist, mirroring the cart's storage architecture
 * (localStorage list of product IDs; see cart.js). There is no wishlist UI
 * beyond the "Move to Wishlist" action on the cart page yet — this module is
 * the single source of truth for saved IDs so that UI can be added later
 * without changing the storage.
 *
 * Primary API is window.SmartRecoWishlist; window.UPulseWishlist is an alias.
 * Mutations fire a 'wishlist' window event so any page can react.
 */
(function () {
    const KEY = 'upulse_wishlist_ids';

    function get() {
        try {
            const raw = localStorage.getItem(KEY);
            const parsed = raw ? JSON.parse(raw) : [];
            return Array.isArray(parsed) ? parsed : [];
        } catch (err) {
            return [];
        }
    }

    function _save(ids) {
        localStorage.setItem(KEY, JSON.stringify(ids));
        window.dispatchEvent(new Event('wishlist'));
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

    const api = { get, add, remove, clear, contains };
    window.SmartRecoWishlist = api;
    window.UPulseWishlist = api;
})();
