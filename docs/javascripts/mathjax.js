window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    typeset: false,
    ready() {
      MathJax.startup.defaultReady();
      // Serialize typesetting after startup, including Material page navigation.
      document$.subscribe(() => {
        MathJax.startup.promise = MathJax.startup.promise.then(() => {
          MathJax.typesetClear();
          MathJax.texReset();
          return MathJax.typesetPromise();
        }).catch((error) => console.error("MathJax typesetting failed:", error));
      });
    }
  }
};
