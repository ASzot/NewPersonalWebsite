
# Setup
- Install hugo with `brew install hugo`
- `git submodule update --init`

# Editing
- The homepage is at `themes/loveit/layouts/partials/home/profile.html`
- The header is at `themes/loveit/layouts/partials/header.html`
- To start the debug server run `hugo server -D`
- If you make changes to the javascript, you have to rebuild, in `themes/loveit` run: `npx babel src/js --out-file assets/js/theme.min.js --source-maps`.
