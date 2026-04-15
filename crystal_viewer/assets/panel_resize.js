(function () {
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function bindSplitter(splitterId, panelId, edge) {
    const splitter = document.getElementById(splitterId);
    const panel = document.getElementById(panelId);
    const root = document.getElementById("viewer-root");
    if (!splitter || !panel || !root || splitter.dataset.bound === "1") {
      return;
    }

    splitter.dataset.bound = "1";
    splitter.addEventListener("mousedown", function (event) {
      event.preventDefault();
      const rootRect = root.getBoundingClientRect();
      document.body.classList.add("panel-resizing");

      function onMove(moveEvent) {
        let width;
        if (edge === "left") {
          width = clamp(moveEvent.clientX - rootRect.left, 260, 640);
        } else {
          width = clamp(rootRect.right - moveEvent.clientX, 260, 640);
        }
        panel.style.width = width + "px";
        panel.style.flex = "0 0 auto";
      }

      function onUp() {
        document.body.classList.remove("panel-resizing");
        window.removeEventListener("mousemove", onMove);
      }

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp, { once: true });
    });
  }

  function init() {
    bindSplitter("left-splitter", "left-panel", "left");
    bindSplitter("right-splitter", "right-panel", "right");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  const observer = new MutationObserver(function () {
    init();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
