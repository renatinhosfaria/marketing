/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    const apiUrl = process.env.API_INTERNAL_URL || 'http://localhost:8000';
    const fbAdsUrl = process.env.FB_ADS_INTERNAL_URL || 'http://localhost:8002';
    const agentUrl = process.env.AGENT_INTERNAL_URL || 'http://localhost:8001';
    return [
      {
        source: '/api/v1/agent/:path*',
        destination: `${agentUrl}/api/v1/agent/:path*`,
      },
      {
        source: '/api/v1/facebook-ads/:path*',
        destination: `${fbAdsUrl}/api/v1/facebook-ads/:path*`,
      },
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
