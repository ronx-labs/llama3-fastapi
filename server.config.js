module.exports = {
    apps: [
        {
            name: "llama3-server",
            script: "/workspace/llama3/venv/bin/uvicorn server:app --reload",
            autorestart: true,
            watch: true,
            env: {
                NODE_ENV: "development",
            },
            env_production: {
                NODE_ENV: "production",
            },
        },
    ],
};
