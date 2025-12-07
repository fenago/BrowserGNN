# Publishing BrowserGNN to npm

This document details how to publish and maintain the BrowserGNN package on npm.

## Quick Reference

| Resource | URL |
|----------|-----|
| **npm Package** | https://www.npmjs.com/package/browser-gnn |
| **GitHub Repository** | https://github.com/fenago/BrowserGNN |
| **Live Demo** | https://browsergnn.com |

---

## Prerequisites

1. **npm account**: Sign up at https://www.npmjs.com/signup
2. **Node.js 18+**: Required for building the package
3. **Git**: For version control

---

## Initial Setup (First Time Only)

### 1. Create npm Account

```bash
npm adduser
```

Or sign up at https://www.npmjs.com/signup

### 2. Login to npm

```bash
npm login
```

You'll be prompted for:
- Username
- Password
- Email
- One-time password (if 2FA is enabled)

### 3. Verify Login

```bash
npm whoami
```

---

## Publishing Process

### Step 1: Ensure Tests Pass

```bash
npm test
```

All 57 tests should pass before publishing.

### Step 2: Build the Package

```bash
npm run build
```

This compiles TypeScript to JavaScript and generates:
- `dist/index.js` (CommonJS)
- `dist/index.esm.js` (ES Modules)
- `dist/index.d.ts` (TypeScript definitions)

### Step 3: Verify Package Contents

```bash
npm pack --dry-run
```

This shows what files will be included:
```
dist/
README.md
LICENSE
```

### Step 4: Publish

```bash
npm publish
```

The package will be live at: https://www.npmjs.com/package/browser-gnn

---

## Version Management

### Semantic Versioning

BrowserGNN follows [semver](https://semver.org/):

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fixes | Patch | 0.1.0 ’ 0.1.1 |
| New features (backward compatible) | Minor | 0.1.0 ’ 0.2.0 |
| Breaking changes | Major | 0.1.0 ’ 1.0.0 |

### Bump Version and Publish

```bash
# Patch release (bug fixes)
npm version patch
npm publish

# Minor release (new features)
npm version minor
npm publish

# Major release (breaking changes)
npm version major
npm publish
```

### Pre-release Versions

```bash
# Alpha release
npm version 1.0.0-alpha.1
npm publish --tag alpha

# Beta release
npm version 1.0.0-beta.1
npm publish --tag beta
```

---

## Testing the Published Package

### Method 1: Install in a New Project

```bash
mkdir test-browser-gnn
cd test-browser-gnn
npm init -y
npm install browser-gnn
```

Create `test.js`:
```javascript
import { GraphData, GCNConv, createBrowserGNN } from 'browser-gnn';

async function test() {
  const { backend } = await createBrowserGNN();
  console.log('BrowserGNN initialized with backend:', backend);

  const graph = new GraphData({
    x: new Float32Array([1, 2, 3, 4, 5, 6]),
    numNodes: 2,
    numFeatures: 3,
    edgeIndex: new Uint32Array([0, 1, 1, 0]),
    numEdges: 2
  });

  console.log('Graph created:', graph.toString());

  const gcn = new GCNConv({ inChannels: 3, outChannels: 4 });
  const output = gcn.forward(graph);
  console.log('GCN output shape:', output.x.shape);
}

test();
```

Run:
```bash
node test.js
```

### Method 2: Test in Browser

Create `test.html`:
```html
<!DOCTYPE html>
<html>
<head>
  <title>BrowserGNN Test</title>
</head>
<body>
  <script type="module">
    import { GraphData, GCNConv, createBrowserGNN } from 'https://unpkg.com/browser-gnn';

    async function test() {
      const { backend } = await createBrowserGNN();
      console.log('BrowserGNN initialized:', backend);

      const graph = new GraphData({
        x: new Float32Array([1, 2, 3, 4]),
        numNodes: 2,
        numFeatures: 2,
        edgeIndex: new Uint32Array([0, 1, 1, 0]),
        numEdges: 2
      });

      console.log('Graph:', graph.toString());
    }

    test();
  </script>
</body>
</html>
```

---

## Package Configuration

### package.json Key Fields

```json
{
  "name": "browser-gnn",
  "version": "0.1.0",
  "description": "BrowserGNN by Dr. Lee - The World's First Comprehensive Graph Neural Network Library for the Browser",
  "author": "Dr. Lee",
  "license": "MIT",
  "main": "dist/index.js",
  "module": "dist/index.esm.js",
  "types": "dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.esm.js",
      "require": "./dist/index.js"
    }
  },
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "repository": {
    "type": "git",
    "url": "https://github.com/fenago/BrowserGNN.git"
  },
  "homepage": "https://github.com/fenago/BrowserGNN",
  "bugs": {
    "url": "https://github.com/fenago/BrowserGNN/issues"
  }
}
```

### Fields Explained

| Field | Purpose |
|-------|---------|
| `main` | Entry point for CommonJS (require) |
| `module` | Entry point for ES Modules (import) |
| `types` | TypeScript definitions |
| `exports` | Modern entry points with conditions |
| `files` | What gets published to npm |
| `prepublishOnly` | Runs build before publishing |

---

## Troubleshooting

### "Package name already exists"

The name `browser-gnn` is registered. If you see this error, you're likely not logged in as the package owner.

### "You must be logged in to publish"

```bash
npm login
```

### "Version already exists"

Bump the version first:
```bash
npm version patch
npm publish
```

### "Missing dist folder"

Build first:
```bash
npm run build
```

### Verify Package Owner

```bash
npm owner ls browser-gnn
```

---

## Unpublishing (Emergency Only)

You can unpublish within 72 hours:

```bash
npm unpublish browser-gnn@0.1.0
```

**Warning**: This breaks dependent projects. Only use for critical issues.

---

## Deprecating Versions

Mark a version as deprecated:

```bash
npm deprecate browser-gnn@0.1.0 "Critical bug, please upgrade to 0.1.1"
```

---

## Useful Commands

```bash
# View package info
npm view browser-gnn

# View all versions
npm view browser-gnn versions

# View package on npm
npm docs browser-gnn

# Check for vulnerabilities
npm audit

# See what will be published
npm pack --dry-run
```

---

## Checklist Before Publishing

- [ ] All tests pass (`npm test`)
- [ ] Build succeeds (`npm run build`)
- [ ] Version bumped appropriately
- [ ] README is up to date
- [ ] CHANGELOG updated (if maintained)
- [ ] No secrets in code
- [ ] `npm pack --dry-run` shows correct files

---

## Links

- **npm Package**: https://www.npmjs.com/package/browser-gnn
- **GitHub**: https://github.com/fenago/BrowserGNN
- **Live Demo**: https://browsergnn.com
- **npm Documentation**: https://docs.npmjs.com/
- **Semantic Versioning**: https://semver.org/
