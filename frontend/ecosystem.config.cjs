module.exports = {
  apps: [{
    name: 'famachat-ml-frontend',
    script: 'node_modules/.bin/next',
    args: 'start -p 3001',
    cwd: '/var/www/famachat-ml/frontend',
    instances: 1,
    exec_mode: 'fork',
    max_memory_restart: '300M',
    restart_delay: 3000,
    out_file: '/var/www/famachat-ml/logs/frontend-out.log',
    error_file: '/var/www/famachat-ml/logs/frontend-error.log',
    env: {
      NODE_ENV: 'production',
      NEXT_PUBLIC_API_URL: '',
    }
  }]
};
