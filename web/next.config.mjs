const origin = process.env.FIELD_API_URL || 'http://127.0.0.1:8787';
export default {
  // Excel uploads pass through this proxy to the Rust API (which accepts up to 64 MB).
  // proxyTimeout: the default 30 s cut long requests (demo build, imports, CRM commit) mid-way; the API kills its worker
  // when the connection drops, which left half-built data. 10 min is above the API's own 300 s limit for these.
  experimental: { proxyClientMaxBodySize: '64mb', proxyTimeout: 600_000 },
  // On Vercel the root vercel.json routes /api/* to the api service before requests reach Next.js.
  async rewrites() {
    return process.env.VERCEL ? [] : [{ source: '/api/:path*', destination: origin + '/api/:path*' }];
  }
};
