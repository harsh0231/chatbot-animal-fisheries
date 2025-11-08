# Deployment Guide - Chatbot Animal & Fisheries

## Quick Start on Render

### Prerequisites
- GitHub account with this repository pushed
- Render account (https://render.com)
- Database access credentials (RDS or MySQL)
- OpenAI API key
- Gemini API key

### Step 1: Connect Repository to Render

1. Go to https://render.com/dashboard
2. Click "New +" → "Web Service"
3. Select "Deploy an existing repository"
4. Connect your GitHub account
5. Select: `harsh0231/chatbot-animal-fisheries`

### Step 2: Configure Web Service

1. **Name**: `chatbot-animal-fisheries`
2. **Environment**: Python
3. **Region**: Choose closest to your data center
4. **Plan**: Standard or Higher (depends on expected traffic)
5. **Build Command**: `pip install -r requirements.txt`
6. **Start Command**: `python -m uvicorn backend:app --host 0.0.0.0 --port $PORT`

### Step 3: Set Environment Variables

In Render Dashboard, add these environment variables:

**Database Configuration:**
- `CHATBOT_DB_HOST`: Your RDS endpoint (e.g., `database.xxxxx.ap-south-1.rds.amazonaws.com`)
- `CHATBOT_DB_USER`: Database username
- `CHATBOT_DB_PASSWORD`: Database password
- `CHATBOT_DB_DATABASE`: Database name (e.g., `bpd_ai_storyboard`)
- `CHATBOT_DB_PORT`: `3306`

**API Keys (KEEP SECURE - Never commit these):**
- `OPENAI_API_KEY`: Your OpenAI API key
- `GEMINI_API_KEY`: Your Google Gemini API key

**Application Configuration:**
- `ALLOWED_ORIGINS`: `*` (or specify allowed domains)
- `ALLOW_CREDENTIALS`: `false`
- `ENV`: `production`

### Step 4: Deploy

1. Click "Create Web Service"
2. Render will automatically deploy from the `render.yaml` file
3. Wait for build and deployment to complete
4. Your chatbot will be available at: `https://<your-service-name>.onrender.com`

## Deployment Checklist

- [ ] Database credentials verified and accessible from Render
- [ ] API keys (OpenAI, Gemini) are valid and have sufficient quota
- [ ] CORS origins configured correctly
- [ ] Health check endpoint working: `/health`
- [ ] Static files (logo.jpg) are accessible: `/static/logo.jpg`
- [ ] Chat endpoint responding: `/chat`
- [ ] Frontend loads without errors

## Monitoring

1. **Logs**: Available in Render Dashboard → Service → Logs
2. **Health Check**: Monitor `/health` endpoint
3. **Alerts**: Set up email notifications for deployment failures

## Production Recommendations

### Security
- [ ] Update CORS `ALLOWED_ORIGINS` to specific domains (not `*`)
- [ ] Enable HTTPS (automatic with Render)
- [ ] Rotate API keys regularly
- [ ] Use environment variables for all secrets
- [ ] Don't commit `.env` file to repository

### Performance
- [ ] Enable database connection pooling (already configured)
- [ ] Configure appropriate log levels (DEBUG → INFO/WARNING for production)
- [ ] Monitor response times
- [ ] Set up caching for repeated queries if needed

### Maintenance
- [ ] Regular database backups
- [ ] Monitor API usage and costs (OpenAI, Gemini)
- [ ] Update dependencies periodically
- [ ] Review error logs weekly

## Troubleshooting

### 502 Bad Gateway
- Check if database is accessible from Render
- Verify environment variables are set correctly
- Check service logs for errors

### "Error: Failed to fetch"
- Ensure CORS is properly configured
- Check if API endpoints are accessible
- Verify database connection in `/health` endpoint

### Slow Response
- Check database query performance
- Monitor API rate limits (OpenAI, Gemini)
- Consider upgrading Render plan

## Database Access

Your application requires access to MySQL database. Ensure:

1. Database accepts connections from Render IPs
2. Security group/firewall allows port 3306 (MySQL)
3. Database credentials are correct
4. Database tables exist and are populated

## Local Testing Before Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
set CHATBOT_DB_HOST=your-database-host
set CHATBOT_DB_USER=your-user
set CHATBOT_DB_PASSWORD=your-password
set CHATBOT_DB_DATABASE=your-database

# Run locally
python -m uvicorn backend:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "language": "english"}'
```

## Useful Render CLI Commands

```bash
# Install Render CLI
npm install -g @render-com/cli

# Login
render login

# Deploy
render deploy --service chatbot-animal-fisheries

# View logs
render logs --service chatbot-animal-fisheries
```

## Support & Documentation

- Render Docs: https://render.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- Uvicorn: https://www.uvicorn.org

---

**Last Updated**: 2025-11-08
**Status**: Ready for Production Deployment
