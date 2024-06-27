import plotly.graph_objs as go

def create_psd_plot(audio_array, sampling_rate):
    from scipy.signal import welch

    freqs, psd = welch(audio_array, sampling_rate)
    psd_plot = go.Figure()
    psd_plot.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD'))
    psd_plot.update_layout(title='Power Spectral Density', xaxis_title='Frequency (Hz)', yaxis_title='Power/Frequency (dB/Hz)')
    return psd_plot

def create_audio_html(index, time_axis, audio_array, real_or_fake, b64_audio, playback_rate, sampling_rate):
    html = f"""
        <div style="margin: 0; padding: 0;">
            <div id="plot_{index}"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    var plotDiv = document.getElementById('plot_{index}');
                    var timeAxis = {time_axis.tolist()};
                    var audioArray = {audio_array.tolist()};

                    Plotly.newPlot(plotDiv, [{{
                        x: timeAxis,
                        y: audioArray,
                        mode: 'lines',
                        name: 'Audio Signal'
                    }}, {{
                        x: [0],
                        y: [audioArray[0]],
                        mode: 'markers',
                        marker: {{color: 'red', size: 10}},
                        name: 'Playback Position'
                    }}], {{
                        title: '{real_or_fake}',
                        showlegend: false,
                        xaxis: {{ title: 'Time (s)' }},
                        yaxis: {{ title: 'Amplitude' }},
                        height: 250,
                        margin: {{ l: 30, r: 10, t: 30, b: 30 }},
                        xaxis: {{ title: {{ standoff: 10 }} }},
                        yaxis: {{ title: {{ standoff: 10 }} }}
                    }});

                    var audio = new Audio('data:audio/wav;base64,{b64_audio}');
                    audio.playbackRate = {playback_rate};

                    var currentMarker = 1;

                    function updateMarker(currentTime) {{
                        var currentSample = Math.floor(currentTime * {sampling_rate});
                        var update = {{
                            x: [[currentTime]],
                            y: [[audioArray[currentSample]]]
                        }};
                        Plotly.restyle(plotDiv, update, [currentMarker]);
                    }}

                    audio.ontimeupdate = function() {{
                        updateMarker(audio.currentTime);
                    }};

                    plotDiv.on('plotly_click', function(data) {{
                        var point = data.points[0];
                        var newTime = point.x;
                        audio.currentTime = newTime;
                        updateMarker(newTime);
                    }});

                    var playButton = document.createElement('button');
                    playButton.innerText = 'Play';
                    playButton.onclick = function() {{
                        if (audio.paused) {{
                            audio.play();
                            playButton.innerText = 'Pause';
                        }} else {{
                            audio.pause();
                            playButton.innerText = 'Play';
                        }}
                    }};
                    
                    audio.onended = function() {{
                        playButton.innerText = 'Play';
                    }};
                    
                    plotDiv.appendChild(playButton);
                }});
            </script>
        </div>
    """
    return html
