(function () {
  var root = document.documentElement;

  function stored() {
    try {
      return localStorage.getItem("theme");
    } catch (e) {
      return null;
    }
  }

  function remember(value) {
    try {
      localStorage.setItem("theme", value);
    } catch (e) {}
  }

  function current() {
    var set = root.getAttribute("data-theme");
    if (set) return set;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  var saved = stored();
  if (saved === "light" || saved === "dark") {
    root.setAttribute("data-theme", saved);
  }

  var button = document.getElementById("theme");
  function label() {
    var dark = current() === "dark";
    button.textContent = dark ? "Light" : "Dark";
    button.setAttribute("aria-label", "Switch to " + (dark ? "light" : "dark") + " theme");
  }
  if (button) {
    label();
    button.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      remember(next);
      label();
    });
  }

  var tip = document.getElementById("tip");
  if (!tip) return;

  function place(x, y) {
    var w = tip.offsetWidth;
    var h = tip.offsetHeight;
    var left = Math.min(Math.max(8, x - w / 2), window.innerWidth - w - 8);
    var top = y - h - 14;
    if (top < 8) top = y + 18;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function show(target, x, y) {
    tip.textContent = target.getAttribute("data-tip");
    tip.classList.add("on");
    place(x, y);
  }

  function hide() {
    tip.classList.remove("on");
  }

  function markOf(node) {
    while (node && node !== document) {
      if (node.getAttribute && node.getAttribute("data-tip")) return node;
      node = node.parentNode;
    }
    return null;
  }

  document.addEventListener("pointerover", function (event) {
    var mark = markOf(event.target);
    if (mark) show(mark, event.clientX, event.clientY);
  });
  document.addEventListener("pointermove", function (event) {
    if (tip.classList.contains("on") && markOf(event.target)) {
      place(event.clientX, event.clientY);
    }
  });
  document.addEventListener("pointerout", function (event) {
    if (markOf(event.target)) hide();
  });
  document.addEventListener("pointerdown", function (event) {
    var mark = markOf(event.target);
    if (mark) show(mark, event.clientX, event.clientY);
    else hide();
  });
  window.addEventListener("scroll", hide, { passive: true });
})();
