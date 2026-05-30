"""
intelligence_tab.py
───────────────────
Drop this entire block inside tab logic of app.py as a new tab:
  "🧠 Intelligence"

Assumes these variables already exist in scope (from app.py):
  df_hist, weather, traffic, firms_df, terrain, pop_m
  ref_pm10, all_stations, selected_city, lat, lon
  elev_df (optional)
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from intelligence import (
    run_forecast_pipeline, attribute_sources,
    detect_events, generate_narrative, EVENT_TYPES,
    run_loocv_calibration, apply_loocv_correction,
)

def render_intelligence_tab(df_hist, weather, traffic, firms_df,
                              terrain, pop_m, ref_pm10,
                              all_stations, selected_city):
    import json

    BG  = "#0e1117"
    BG2 = "#1a1d27"
    ACC = "#ff6b35"

    st.markdown(f"### 🧠 Predictive Intelligence — {selected_city}")

    # ── Run models ──
    with st.spinner("🤖 Training temporal models (XGBoost · LSTM · Prophet · RF)..."):
        pipeline = run_forecast_pipeline(
            df_hist, weather, traffic, firms_df, horizon_h=72)

    forecast_df = pipeline.get("forecast_df", pd.DataFrame())
    fi          = pipeline.get("feature_importance", {})
    decomp      = pipeline.get("decomposition", {})
    trained     = pipeline.get("trained", False)

    # ── LOOCV Calibration ──
    with st.spinner("🔬 Running LOOCV bias calibration..."):
        loocv = run_loocv_calibration(df_hist, weather, traffic)

    # Apply bias correction to forecast
    if not forecast_df.empty and loocv.get("calibrated"):
        forecast_df = apply_loocv_correction(forecast_df, loocv)

    # ── Attribution ──
    attribution = attribute_sources(
        ref_pm10, weather, traffic, firms_df, terrain, pop_m, fi)

    # ── Events ──
    events = detect_events(df_hist, weather, traffic, firms_df, forecast_df)

    # ── Narrative ──
    narrative = generate_narrative(
        ref_pm10, attribution, events, weather, traffic, firms_df, terrain, forecast_df)

    # ════════════════════════════════════════
    # ROW 1: Narrative + Live Alerts
    # ════════════════════════════════════════
    col_narr, col_alerts = st.columns([3, 2])

    with col_narr:
        st.markdown("#### 📋 Automated Cause Analysis")
        for para in narrative.split("\n\n"):
            st.markdown(para)

    with col_alerts:
        st.markdown("#### 🚨 Live Event Alerts")
        if not events:
            st.success("✅ No anomalous events detected. Air quality conditions are within normal range.")
        else:
            for ev in events[:6]:
                sev = ev.get("severity", "low")
                etype = ev.get("type", "pm_spike")
                einfo = EVENT_TYPES.get(etype, {})
                icon  = einfo.get("icon", "⚠️")
                clr   = einfo.get("color", "#ffcc00")
                bg    = {"high":"rgba(255,0,0,0.1)",
                          "medium":"rgba(255,126,0,0.1)",
                          "low":"rgba(61,90,128,0.1)"}.get(sev, "rgba(50,50,80,0.1)")
                border= {"high":"#ff0000","medium":"#ff7e00","low":"#3d5a80"}.get(sev,"#888")
                st.markdown(f"""
                <div style='background:{bg};border-left:3px solid {border};
                  padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px'>
                  <div style='font-weight:600;color:{clr};font-size:14px'>
                    {icon} {ev["title"]}
                  </div>
                  <div style='font-size:12px;color:#ccc;margin-top:4px'>
                    {ev["detail"]}
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════
    # ROW 2: 72-Hour Forecast + Model Bands
    # ════════════════════════════════════════
    st.markdown("#### 📈 72-Hour Ensemble Forecast with Uncertainty")

    if trained and not forecast_df.empty:
        mq = pipeline.get("model_quality", {})
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("XGBoost",  "✅ Trained" if mq.get("xgb_trained")     else "⚠️ Fallback")
        mc2.metric("LSTM",     "✅ Trained" if mq.get("lstm_trained")     else "⚠️ Fallback")
        mc3.metric("Prophet",  "✅ Trained" if mq.get("prophet_trained")  else "⚠️ Fallback")
        mc4.metric("RF Ensemble","✅ Trained" if mq.get("rf_trained")     else "⚠️ Fallback")

        fig_fc = go.Figure()

        # Confidence band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast_df["timestamp"], forecast_df["timestamp"][::-1]]),
            y=pd.concat([forecast_df["pm10_upper"], forecast_df["pm10_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,107,53,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% confidence band",
            hoverinfo="skip",
        ))

        # Per-model lines (thin, background)
        for col_, color_, name_ in [
            ("xgb",     "rgba(255,107,53,0.45)", "XGBoost"),
            ("rf",      "rgba(61,90,128,0.45)",  "RF"),
            ("prophet", "rgba(100,200,100,0.45)","Prophet"),
            ("lstm",    "rgba(200,150,255,0.45)","LSTM"),
        ]:
            if col_ in forecast_df.columns:
                fig_fc.add_trace(go.Scatter(
                    x=forecast_df["timestamp"], y=forecast_df[col_],
                    mode="lines", name=name_,
                    line=dict(color=color_, width=1, dash="dot"),
                    hovertemplate=f"{name_}: %{{y:.0f}} µg/m³<extra></extra>",
                ))

        # Ensemble mean (bold)
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["pm10_mean"],
            mode="lines", name="Ensemble mean",
            line=dict(color=ACC, width=3),
            hovertemplate="Ensemble: %{y:.0f} µg/m³<extra></extra>",
        ))

        # AQI threshold bands
        for level, clr, lbl in [
            (50,"#00e400","Good"), (100,"#ffff00","Moderate"),
            (250,"#ff7e00","Poor"), (350,"#ff0000","Very Poor")]:
            fig_fc.add_hline(y=level, line_dash="dot", line_color=clr,
                              line_width=0.8, annotation_text=lbl,
                              annotation_position="right",
                              annotation_font_color=clr)

        # 24h / 48h dividers
        now = pd.Timestamp.now()
        for h, lbl in [(24,"24h"), (48,"48h")]:
            _vx = str((now + pd.Timedelta(hours=h)).isoformat())
            fig_fc.add_shape(type="line", xref="x", yref="paper",
                              x0=_vx, x1=_vx, y0=0, y1=1,
                              line=dict(dash="dash", color="rgba(255,255,255,0.2)", width=1))
            fig_fc.add_annotation(x=_vx, yref="paper", y=1.02,
                                   text=lbl, showarrow=False,
                                   font=dict(color="rgba(255,255,255,0.5)", size=11))

        fig_fc.update_layout(
            xaxis_title="Time", yaxis_title="PM10 (µg/m³)",
            paper_bgcolor=BG, plot_bgcolor=BG2, font_color="white",
            legend=dict(bgcolor=BG2, bordercolor="#333"),
            height=420, margin=dict(t=20,b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption("Shaded band = 90% confidence interval. Thin dotted lines = individual model predictions.")

        # Fire-triggered spike simulation
        if not firms_df.empty:
            st.markdown("##### 🔥 What-if: fire plume arrives in 4 hours")
            fire_mult = st.slider("Fire intensity multiplier", 1.0, 3.0, 1.5, 0.1)
            # Apply multiplier to forecast from hour 4 onward
            fc_fire = forecast_df.copy()
            fc_fire.loc[fc_fire.index >= 4, "pm10_mean"] *= fire_mult
            fc_fire["pm10_mean"] = fc_fire["pm10_mean"].clip(0, 600)
            fig_fire = go.Figure()
            fig_fire.add_trace(go.Scatter(x=forecast_df["timestamp"],
                y=forecast_df["pm10_mean"], name="Base forecast",
                line=dict(color=ACC, width=2)))
            fig_fire.add_trace(go.Scatter(x=fc_fire["timestamp"],
                y=fc_fire["pm10_mean"], name="With fire plume",
                line=dict(color="#ff4500", width=2, dash="dash")))
            fig_fire.add_hline(y=250, line_dash="dot", line_color="#ff7e00",
                                annotation_text="Poor threshold")
            fig_fire.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2,
                font_color="white", height=300, margin=dict(t=10,b=30))
            st.plotly_chart(fig_fire, use_container_width=True)
            st.caption("Drag slider to simulate different fire intensities. "
                        "Crossing 250 µg/m³ = 'Poor' AQ — health impacts begin.")

    else:
        st.info("Insufficient historical data to train temporal models. "
                "The system needs ≥24 timestamped records. "
                "Run the app daily to build the dataset.")
        # Show physics-based simple forecast instead
        if not weather.get("forecast_df", pd.DataFrame()).empty:
            st.markdown("*Showing physics-based estimate while data accumulates:*")

    st.markdown("---")

    # ════════════════════════════════════════
    # LOOCV CALIBRATION REPORT
    # ════════════════════════════════════════
    st.markdown("#### 🔬 LOOCV Bias Calibration — Model Self-Correction")

    if loocv.get("calibrated"):
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        bias_color = "#ff6b35" if abs(loocv["bias"]) > 10 else "#00e400"
        lc1.metric("Systematic Bias",
                   f"{loocv['bias']:+.1f} µg/m³",
                   delta="corrected ✓" if loocv["calibrated"] else None)
        lc2.metric("Bias %",        f"{loocv['bias_pct']:+.1f}%")
        lc3.metric("MAE",           f"{loocv['mae']:.2f} µg/m³")
        lc4.metric("RMSE",          f"{loocv['rmse']:.2f} µg/m³")
        lc5.metric("Error Std σ",   f"{loocv['error_std']:.2f} µg/m³")

        # Bias interpretation
        bias = loocv["bias"]
        if abs(bias) < 3:
            st.success(f"✅ Model is well-calibrated. Bias of {bias:+.1f} µg/m³ is within acceptable range. No significant correction needed.")
        elif bias > 0:
            st.warning(f"⚠️ Model **underpredicts** by {bias:.1f} µg/m³ on average. "
                        f"All forecasts have been shifted up by {bias:.1f} µg/m³ to correct this.")
        else:
            st.warning(f"⚠️ Model **overpredicts** by {abs(bias):.1f} µg/m³ on average. "
                        f"All forecasts have been shifted down by {abs(bias):.1f} µg/m³ to correct this.")

        # Residual distribution plot
        residuals = loocv.get("residuals", np.array([]))
        if len(residuals) > 0:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                fig_res = go.Figure()
                fig_res.add_trace(go.Histogram(
                    x=residuals, nbinsx=20,
                    marker_color="#3d5a80",
                    name="Residuals",
                ))
                fig_res.add_vline(x=0, line_dash="dash", line_color="white",
                                   annotation_text="Zero bias")
                fig_res.add_vline(x=float(np.mean(residuals)),
                                   line_dash="dot", line_color="#ff6b35",
                                   annotation_text=f"Mean={np.mean(residuals):.1f}")
                fig_res.update_layout(
                    title="Residual Distribution (Actual − Predicted)",
                    xaxis_title="Residual (µg/m³)",
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
                    font_color="white", height=280, margin=dict(t=40,b=30))
                st.plotly_chart(fig_res, use_container_width=True)

            with col_res2:
                if loocv.get("actuals") is not None and loocv.get("preds") is not None:
                    actuals_arr = loocv["actuals"]
                    preds_arr   = loocv["preds"]
                    # Corrected preds
                    preds_corr  = preds_arr + loocv["correction_offset"]
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Scatter(
                        x=actuals_arr, y=preds_arr,
                        mode="markers", name="Raw predictions",
                        marker=dict(color="#888", size=7, opacity=0.6)))
                    fig_cv.add_trace(go.Scatter(
                        x=actuals_arr, y=preds_corr,
                        mode="markers", name="Bias-corrected",
                        marker=dict(color="#ff6b35", size=7)))
                    # Perfect line
                    mn = min(actuals_arr.min(), preds_arr.min())
                    mx = max(actuals_arr.max(), preds_arr.max())
                    fig_cv.add_trace(go.Scatter(
                        x=[mn, mx], y=[mn, mx], mode="lines",
                        line=dict(dash="dash", color="white", width=1),
                        name="Perfect fit"))
                    fig_cv.update_layout(
                        title="LOOCV: Actual vs Predicted",
                        xaxis_title="Actual PM10", yaxis_title="Predicted PM10",
                        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
                        font_color="white", height=280, margin=dict(t=40,b=30))
                    st.plotly_chart(fig_cv, use_container_width=True)
                    st.caption(f"Grey = raw model · Orange = bias-corrected · "
                                f"Correction applied: {loocv['correction_offset']:+.1f} µg/m³")

        # Per-station errors
        if loocv.get("per_station"):
            st.markdown("**Per-station bias:**")
            ps_df = pd.DataFrame([
                {"Location": k, "Bias (µg/m³)": v["bias"],
                 "MAE": v["mae"], "N readings": v["n"]}
                for k, v in loocv["per_station"].items()
            ])
            st.dataframe(ps_df, use_container_width=True)
    else:
        st.info("LOOCV requires ≥4 sensor readings. "
                "Select a city with monitoring stations (Delhi, Mumbai, etc.) "
                "and accumulate historical data.")

    st.markdown("---")

    # ════════════════════════════════════════
    # ROW 3: Source Attribution
    # ════════════════════════════════════════
    st.markdown("#### 🔬 Source Attribution — PM10 Breakdown")

    col_donut, col_bars = st.columns([1, 2])

    with col_donut:
        labels = [attribution[k]["label"].split("(")[0].strip() for k in attribution]
        values = [attribution[k]["µg/m³"] for k in attribution]
        colors = [attribution[k]["color"] for k in attribution]
        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color="#1a1d27", width=2)),
            hole=0.55,
            textinfo="percent",
            hovertemplate="%{label}: %{value:.1f} µg/m³ (%{percent})<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{ref_pm10:.0f}</b><br>µg/m³",
            x=0.5, y=0.5, font_size=18, showarrow=False,
            font=dict(color="white"),
        )
        fig_donut.update_layout(
            paper_bgcolor=BG, font_color="white",
            showlegend=False, height=280, margin=dict(t=10,b=10,l=10,r=10),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bars:
        attr_df = pd.DataFrame([
            {"Source": attribution[k]["label"].split("(")[0].strip(),
             "µg/m³": attribution[k]["µg/m³"],
             "Color": attribution[k]["color"],
             "Pct": attribution[k]["pct"]}
            for k in attribution
        ]).sort_values("µg/m³", ascending=True)

        fig_attr = go.Figure()
        for _, row in attr_df.iterrows():
            fig_attr.add_trace(go.Bar(
                y=[row["Source"]], x=[row["µg/m³"]],
                orientation="h",
                marker_color=row["Color"],
                name=row["Source"],
                text=f"{row['µg/m³']:.1f} µg/m³ ({row['Pct']:.0f}%)",
                textposition="outside",
                hovertemplate="%{text}<extra></extra>",
                width=0.6,
            ))
        fig_attr.update_layout(
            xaxis_title="µg/m³",
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font_color="white", showlegend=False,
            height=280, margin=dict(t=10,b=30,l=10,r=80),
            barmode="overlay",
        )
        st.plotly_chart(fig_attr, use_container_width=True)

    st.markdown("---")

    # ════════════════════════════════════════
    # ROW 4: Seasonal Decomposition
    # ════════════════════════════════════════
    if decomp and len(decomp) == 4:
        st.markdown("#### 📊 Signal Decomposition (Trend · Diurnal · Weekly · Residual)")
        df_hist_sorted = df_hist.sort_values("timestamp").copy() if not df_hist.empty else None

        if df_hist_sorted is not None and not df_hist_sorted.empty:
            n = min(len(decomp["trend"]), len(df_hist_sorted))
            ts = df_hist_sorted["timestamp"].values[:n]

            fig_dec = make_subplots(rows=4, cols=1, shared_xaxes=True,
                subplot_titles=["Trend","Diurnal cycle","Weekly cycle","Residual"],
                vertical_spacing=0.06)

            for i, (comp_name, comp_color) in enumerate([
                ("trend","#ff6b35"),("diurnal","#3d5a80"),
                ("weekly","#1D9E75"),("residual","#888780")], 1):
                arr = decomp[comp_name][:n]
                fig_dec.add_trace(go.Scatter(
                    x=ts, y=arr, mode="lines",
                    line=dict(color=comp_color, width=1.5),
                    name=comp_name.title(),
                    showlegend=False,
                ), row=i, col=1)

            fig_dec.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG2,
                font_color="white", height=480,
                margin=dict(t=30,b=20),
            )
            st.plotly_chart(fig_dec, use_container_width=True)
            st.caption("Trend = long-term direction · Diurnal = daily cycle (rush hours etc.) · "
                        "Weekly = weekday/weekend pattern · Residual = unexplained variance (events, fires, anomalies)")

    st.markdown("---")

    # ════════════════════════════════════════
    # ROW 5: Feature Importance
    # ════════════════════════════════════════
    if fi:
        st.markdown("#### 🏆 What drives PM10 most? (XGBoost feature importance)")
        fi_df = (pd.DataFrame({"Feature": list(fi.keys()),
                                "Importance": list(fi.values())})
                   .sort_values("Importance", ascending=False)
                   .head(15))

        # Group into categories for color coding
        def feat_cat(f):
            if "lag" in f or "roll" in f or "delta" in f: return "#ff6b35"
            if any(x in f for x in ["wind","pressure","humid","trap","temp"]): return "#3d5a80"
            if "fire" in f or "congestion" in f: return "#ff4500"
            return "#888780"

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker_color=[feat_cat(f) for f in fi_df["Feature"]],
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig_fi.update_layout(
            xaxis_title="Importance", yaxis_title="",
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font_color="white", height=420,
            margin=dict(t=10,b=30,l=10,r=60),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # Legend
        st.markdown("""
        <div style='display:flex;gap:16px;font-size:12px;flex-wrap:wrap'>
          <span><span style='color:#ff6b35'>■</span> Temporal lags & rolling stats</span>
          <span><span style='color:#3d5a80'>■</span> Meteorological</span>
          <span><span style='color:#ff4500'>■</span> Emission sources (fire, traffic)</span>
          <span><span style='color:#888780'>■</span> Calendar features</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════
    # ROW 6: Anomaly Timeline
    # ════════════════════════════════════════
    if not df_hist.empty and "pm10" in df_hist.columns and len(df_hist) >= 12:
        st.markdown("#### 📉 Anomaly Timeline (z-score vs 7-day baseline)")
        df_anom = df_hist.sort_values("timestamp").copy()
        df_anom["rolling_mean"] = df_anom["pm10"].rolling(168, min_periods=12).mean()
        df_anom["rolling_std"]  = df_anom["pm10"].rolling(168, min_periods=12).std().clip(lower=1)
        df_anom["z_score"]      = (df_anom["pm10"] - df_anom["rolling_mean"]) / df_anom["rolling_std"]

        fig_anom = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=["PM10 with anomalies", "Z-score"],
            vertical_spacing=0.1)

        fig_anom.add_trace(go.Scatter(x=df_anom["timestamp"], y=df_anom["pm10"],
            mode="lines", line=dict(color="#888", width=1), name="PM10"), row=1, col=1)

        anomalies = df_anom[df_anom["z_score"].abs() > 2]
        if not anomalies.empty:
            fig_anom.add_trace(go.Scatter(
                x=anomalies["timestamp"], y=anomalies["pm10"],
                mode="markers", marker=dict(color="#ff0000", size=8, symbol="circle"),
                name="Anomaly (|z|>2)"), row=1, col=1)

        fig_anom.add_trace(go.Scatter(x=df_anom["timestamp"], y=df_anom["z_score"],
            mode="lines", line=dict(color="#3d5a80", width=1.5), name="z-score"), row=2, col=1)
        fig_anom.add_hline(y=2,  line_dash="dot", line_color="#ff0000", row=2, col=1)
        fig_anom.add_hline(y=-2, line_dash="dot", line_color="#ff0000", row=2, col=1)
        fig_anom.add_hline(y=0,  line_dash="solid", line_color="rgba(255,255,255,0.1)", row=2, col=1)

        fig_anom.update_layout(paper_bgcolor=BG, plot_bgcolor=BG2,
            font_color="white", height=400, margin=dict(t=30,b=20))
        st.plotly_chart(fig_anom, use_container_width=True)
        st.caption("Red dots = anomalous readings (|z|>2). Z-score bands at ±2σ define event threshold.")

        # Anomaly count summary
        n_anomalies = len(anomalies)
        if n_anomalies > 0:
            st.warning(f"⚠️ {n_anomalies} anomalous readings detected in historical data. "
                        f"Most recent: {anomalies['timestamp'].max().strftime('%d %b %H:%M') if not anomalies.empty else 'N/A'}")
        else:
            st.success("✅ No anomalies detected in historical record.")

    # ════════════════════════════════════════
    # ROW 7: Model health
    # ════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### ⚙️ Intelligence System Status")
    s1,s2,s3,s4,s5 = st.columns(5)
    data_pts = len(df_hist) if not df_hist.empty else 0
    s1.metric("Historical records", data_pts)
    s2.metric("Models trained", sum(1 for v in pipeline.get("model_quality",{}).values() if v))
    s3.metric("Events detected", len(events))
    s4.metric("Forecast horizon", "72h" if trained else "N/A")
    s5.metric("Attribution sources", len(attribution))

    progress_val = min(data_pts / 500, 1.0)
    st.progress(progress_val, text=f"Data maturity: {data_pts}/500 records "
                f"({'Full intelligence active' if data_pts>=500 else 'Growing — run daily to improve accuracy'})")
