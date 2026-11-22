const origin = process.env.FIELD_API_URL || 'http://127.0.0.1:8787';
export default {
  async rewrites() {
    return [{ source: '/api/:path*', destination: origin + '/api/:path*' }];
  }
};
