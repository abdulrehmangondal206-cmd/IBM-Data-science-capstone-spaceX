-- Load the official dataset into a table named spacex.
-- Launch counts by site
SELECT LaunchSite, COUNT(*) AS launches FROM spacex GROUP BY LaunchSite ORDER BY launches DESC;
-- Launches by orbit
SELECT Orbit, COUNT(*) AS launches FROM spacex GROUP BY Orbit ORDER BY launches DESC;
-- Average payload by orbit
SELECT Orbit, ROUND(AVG(PayloadMass),2) AS avg_payload_kg FROM spacex GROUP BY Orbit ORDER BY avg_payload_kg DESC;
-- Landing success rate by site
SELECT LaunchSite, COUNT(*) AS total_launches, SUM(Class) AS successful_landings, ROUND(100.0*SUM(Class)/COUNT(*),2) AS success_rate_pct FROM spacex GROUP BY LaunchSite ORDER BY success_rate_pct DESC;
-- Yearly trend (SQLite)
SELECT strftime('%Y',Date) AS launch_year, COUNT(*) AS launches, SUM(Class) AS successful_landings FROM spacex GROUP BY launch_year ORDER BY launch_year;
