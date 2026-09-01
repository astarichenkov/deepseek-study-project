# Deployment Guide — clean Ubuntu VPS (Docker Compose)

This guide deploys the DeepSeek Study App to a fresh Ubuntu server using
Docker and Docker Compose. Target architecture:

```
Internet -> Nginx :80 -> FastAPI :8000 (inside Docker network)
```

The FastAPI port `8000` is **not** exposed publicly — only Nginx `:80` is.

---

## 1. Install Docker

```bash
# Update package index and install prerequisites
sudo apt update
sudo apt install -y ca-certificates curl

# Add Docker's official GPG key and repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine + CLI
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add your user to the docker group (re-login afterwards)
sudo usermod -aG docker "$USER"
```

Log out and back in (or run `newgrp docker`) so the group takes effect.

## 2. Install the Docker Compose plugin

```bash
sudo apt install -y docker-compose-plugin
docker compose version   # expect: Docker Compose version v2.x
```

## 3. Clone the repository

```bash
git clone <your-repository-url> deepseek-study-app
cd deepseek-study-app
```

## 4. Set the DEEPSEEK_API_KEY securely

```bash
cp .env.example .env
# The key is stored ONLY on the server, in a git-ignored file:
#   DEEPSEEK_API_KEY=sk-your-real-key
nano .env
chmod 600 .env
```

Verify the file is ignored by git:

```bash
git status --short   # .env must NOT appear as untracked/changeable
git check-ignore .env && echo ".env is ignored"
```

## 5. Start the containers

```bash
docker compose up -d --build
```

## 6. Check container status

```bash
docker compose ps
```

Both `app` and `nginx` should be `Up` and healthy.

## 7. Check logs

```bash
docker compose logs -f app     # FastAPI logs (Ctrl+C to exit)
docker compose logs -f nginx   # Nginx access/error logs
```

## 8. Test /health

```bash
curl -s http://localhost/health
# {"status":"ok","application":"deepseek-study-app"}
```

You can also test the real chat endpoint (this consumes a tiny amount of
balance):

```bash
curl -s -X POST http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Say OK"}'
```

Open `http://<server-ip>/` in a browser. If you have a domain, point it at
the server and set `server_name` in `nginx/nginx.conf` (optionally add
TLS with certbot).

## 9. Restarting

```bash
docker compose restart          # restart containers (no rebuild)
docker compose down             # stop and remove containers
docker compose up -d            # start again
```

## 10. Updating the application

```bash
cd deepseek-study-app
git pull                        # pull the latest code
docker compose up -d --build    # rebuild image + restart
docker compose ps               # verify
curl -s http://localhost/health
```

## 11. Rollback considerations

- Docker Compose keeps the previous image layers locally, but the `latest`
  tag is overwritten on rebuild. For reliable rollback, tag releases:

  ```bash
  docker compose build
  docker tag deepseek-study-app:latest deepseek-study-app:release-2025-01-01
  ```

  To roll back to a known-good tag:

  ```bash
  docker compose down
  # edit docker-compose.yml: image: deepseek-study-app:release-2025-01-01
  docker compose up -d
  ```

- Before upgrading, check the new commit for breaking changes
  (`git log --oneline -5`, `git diff HEAD~1`).
- The `.env` file survives `git pull` (it is git-ignored), so the API key
  is never lost or overwritten by updates.
- If a deployment breaks, stop the new containers and re-run the previous
  image: `docker compose down && docker compose up -d` (after restoring
  the previous tag in `docker-compose.yml`).
- Consider snapshotting the VPS (cloud provider snapshot / `doctl` /
  AWS AMI) before major upgrades.

---

## Troubleshooting

| Symptom                              | Likely fix                                   |
|--------------------------------------|----------------------------------------------|
| `401` from `/api/chat`               | Wrong/expired `DEEPSEEK_API_KEY` in `.env`; `docker compose up -d` again |
| `502` from `/api/chat`               | Network/firewall blocks outbound HTTPS, or DeepSeek is unreachable |
| Port 80 already in use               | `sudo lsof -i :80` to find the process; free the port |
| `docker compose` not found           | Install the compose plugin (section 2)       |
| Nginx 502 on `/health`               | `docker compose logs app`; app crashed?      |
