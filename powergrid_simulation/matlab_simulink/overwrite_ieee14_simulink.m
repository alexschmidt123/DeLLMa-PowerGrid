function overwrite_ieee14_simulink()
%OVERWRITE_IEEE14_SIMULINK  Apply intial_settings.json, run Simulink model, write dataset.csv.
%
%   Command line (macOS):
%     matlab -batch "cd('/path/to/matlab_simulink'); overwrite_ieee14_simulink; exit"
%
%   JSON drives: model path, solver/stop time, base workspace overrides for masked blocks,
%   and CSV output. Edit intial_settings.json for your research (no need to edit this file).

    root = fileparts(mfilename('fullpath'));
    cfgPath = fullfile(root, 'intial_settings.json');
    if ~isfile(cfgPath)
        error('overwrite_ieee14_simulink:MissingConfig', 'Expected %s', cfgPath);
    end
    cfg = jsondecode(fileread(cfgPath));

    apply_workspace_from_config(cfg);

    mdlPath = fullfile(root, char(cfg.MODEL_REL_PATH));
    if ~isfile(mdlPath)
        error('overwrite_ieee14_simulink:MissingModel', 'Model not found: %s', mdlPath);
    end
    if ~license('test', 'Simulink')
        error('overwrite_ieee14_simulink:NoSimulink', 'Simulink license not available.');
    end

    [~, mdlName, ext] = fileparts(mdlPath);
    if isempty(ext)
        error('overwrite_ieee14_simulink:BadModelPath', 'MODEL_REL_PATH must end in .mdl or .slx');
    end

    load_system(mdlPath);

    set_param(mdlName, 'StopTime', num2str(double(cfg.STOP_TIME_S)));
    if isfield(cfg, 'SOLVER_NAME')
        set_param(mdlName, 'SolverName', char(cfg.SOLVER_NAME));
    end
    if isfield(cfg, 'REL_TOL')
        set_param(mdlName, 'RelTol', char(string(cfg.REL_TOL)));
    end
    if isfield(cfg, 'ABS_TOL')
        set_param(mdlName, 'AbsTol', char(string(cfg.ABS_TOL)));
    end
    if isfield(cfg, 'MAX_STEP_S')
        set_param(mdlName, 'MaxStep', char(string(cfg.MAX_STEP_S)));
    end
    if isfield(cfg, 'MIN_STEP_S')
        set_param(mdlName, 'MinStep', char(string(cfg.MIN_STEP_S)));
    end

    try
        set_param(mdlName, 'LimitDataPoints', 'off');
    catch
    end

    % R2025b: tout only exists if time is saved; isprop(simOut,'tout') is unreliable for SimulationOutput.
    if ~isfield(cfg, 'SAVE_TIME') || cfg.SAVE_TIME
        set_param(mdlName, 'SaveTime', 'on');
        set_param(mdlName, 'TimeSaveName', 'tout');
    end
    if isfield(cfg, 'SIGNAL_LOGGING')
        if cfg.SIGNAL_LOGGING
            set_param(mdlName, 'SignalLogging', 'on');
        else
            set_param(mdlName, 'SignalLogging', 'off');
        end
    end

    % Stock Fourteen_bus has ReturnWorkspaceOutputs off; sim() then omits tout in SimulationOutput.
    if ~isfield(cfg, 'RETURN_WORKSPACE_OUTPUTS') || cfg.RETURN_WORKSPACE_OUTPUTS
        set_param(mdlName, 'ReturnWorkspaceOutputs', 'on');
    else
        set_param(mdlName, 'ReturnWorkspaceOutputs', 'off');
    end

    enable_signal_logging_on_outports(mdlName, cfg);

    fprintf('Running %s (StopTime=%g s, F0_HZ=%g) ...\n', mdlName, cfg.STOP_TIME_S, cfg.F0_HZ);
    simOut = sim(mdlName);
    fprintf('Simulation finished.\n');

    outCsv = fullfile(root, char(cfg.OUTPUT_CSV));
    pref = '';
    if isfield(cfg, 'PREFERRED_LOG_SIGNAL')
        pref = char(cfg.PREFERRED_LOG_SIGNAL);
    end
    write_dataset_from_simout(simOut, cfg, outCsv, pref);
    fprintf('Wrote %s\n', outCsv);

    try
        set_param(mdlName, 'Dirty', 'off');
    catch
    end
    try
        close_system(mdlName, 0);
    catch
    end
end

function enable_signal_logging_on_outports(mdlName, cfg)
    % R2025b: log signals via Outport DataLogging (not line DataLogging).
    if isfield(cfg, 'LOG_OUTPORT1_BLOCKS') && ~isempty(cfg.LOG_OUTPORT1_BLOCKS)
        bl = cfg.LOG_OUTPORT1_BLOCKS;
    else
        bl = {'Bus 1 '};
    end
    if isstring(bl)
        bl = cellstr(bl);
    elseif ischar(bl)
        bl = {bl};
    end
    for i = 1:numel(bl)
        % Do not strtrim: Simulink block names include a trailing space (e.g. "Bus 1 ").
        bn = char(bl{i});
        blk = [mdlName '/' bn];
        try
            get_param(blk, 'BlockType');
        catch
            fprintf(2, 'Warning: LOG_OUTPORT1_BLOCKS block not found: %s\n', blk);
            continue;
        end
        ph = get_param(blk, 'PortHandles');
        if isempty(ph.Outport)
            continue;
        end
        set_param(ph.Outport(1), 'DataLogging', 'on');
        fprintf('Enabled DataLogging on Outport 1 of %s\n', blk);
    end
end

function apply_workspace_from_config(cfg)
    assignin('base', 'F0_HZ', double(cfg.F0_HZ));
    assignin('base', 'STOP_TIME_S', double(cfg.STOP_TIME_S));
    if isfield(cfg, 'WORKSPACE_OVERRIDES') && isstruct(cfg.WORKSPACE_OVERRIDES)
        fo = cfg.WORKSPACE_OVERRIDES;
        names = fieldnames(fo);
        for i = 1:numel(names)
            nm = names{i};
            assignin('base', nm, fo.(nm));
        end
        fprintf('Applied %d WORKSPACE_OVERRIDES from intial_settings.json\n', numel(names));
    end
end

function write_dataset_from_simout(simOut, cfg, csvPath, preferredSubstring)
    if nargin < 4
        preferredSubstring = '';
    end
    f0_hz = double(cfg.F0_HZ);
    tout = extract_tout(simOut);
    if isempty(tout)
        error('write_dataset_from_simout:EmptyTout', ...
            'No time vector (enable SAVE_TIME in JSON and SaveTime in the model).');
    end
    tout = tout(:);

    freq = f0_hz * ones(size(tout));
    rocof = zeros(size(tout));
    usedLogs = false;
    voltageMode = false;
    vpu = nan(size(tout));
    dvpu = nan(size(tout));

    logs = extract_logsout(simOut);
    if ~isempty(logs)
        if isa(logs, 'Simulink.SimulationData.Dataset') && logs.numElements >= 1
            idx = pick_log_element(logs, preferredSubstring);
            if ismethod(logs, 'getElement')
                el = logs.getElement(idx);
            else
                el = logs{idx};
            end
            ts = el.Values;
            if ~isa(ts, 'timeseries')
                error('write_dataset_from_simout:NotTS', 'logsout element is not timeseries.');
            end
            tq = ts.Time(:);
            y_series = scalar_series_from_timeseries(ts);
            if numel(tq) ~= numel(y_series)
                error('write_dataset_from_simout:BadTS', 'Time/Data length mismatch in logsout.');
            end
            usedLogs = true;
            if is_likely_voltage_pu(y_series)
                voltageMode = true;
                vpu = interp1(tq, y_series, tout, 'linear', 'extrap');
                dvpu = gradient(vpu, tout);
                freq = f0_hz * ones(size(tout));
                rocof = zeros(size(tout));
                fprintf(2, '%s\n', 'Note: logged signal looks like per-unit voltage (Three-Phase VI). freq_hz/rocof_hz_s are nominal; use v_pu_rms/dv_pu_dt_s for dynamics.');
            else
                hz = y_to_hz(y_series, f0_hz);
                freq = interp1(tq, hz, tout, 'linear', 'extrap');
                rocof = gradient(freq, tout);
            end
        end
    end

    if ~usedLogs
        fprintf(2, '%s\n', 'Warning: logsout empty; freq_hz set to F0_HZ. Log a frequency/speed signal in Simulink for a dynamic trace.');
        rocof = gradient(freq, tout);
    end

    d = fileparts(csvPath);
    if ~isempty(d) && ~isfolder(d)
        mkdir(d);
    end
    fid = fopen(csvPath, 'w');
    if fid < 0
        error('write_dataset_from_simout:Open', 'Could not open %s', csvPath);
    end
    includeVcols = ~isfield(cfg, 'CSV_INCLUDE_VOLTAGE_COLUMNS') || cfg.CSV_INCLUDE_VOLTAGE_COLUMNS;

    if isfield(cfg, 'CSV_APPEND_METADATA_COLUMNS') && cfg.CSV_APPEND_METADATA_COLUMNS
        if includeVcols
            fprintf(fid, 't,freq_hz,rocof_hz_s,v_pu_rms,dv_pu_dt_s,f0_hz,model_rel_path,stop_time_s\n');
        else
            fprintf(fid, 't,freq_hz,rocof_hz_s,f0_hz,model_rel_path,stop_time_s\n');
        end
        mrp = char(cfg.MODEL_REL_PATH);
        mrp_esc = strrep(mrp, '"', '""');
        sts = double(cfg.STOP_TIME_S);
        for k = 1:numel(tout)
            if includeVcols
                fprintf(fid, '%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"%s",%.12g\n', ...
                    tout(k), freq(k), rocof(k), vpu(k), dvpu(k), f0_hz, mrp_esc, sts);
            else
                fprintf(fid, '%.12g,%.12g,%.12g,%.12g,"%s",%.12g\n', tout(k), freq(k), rocof(k), f0_hz, mrp_esc, sts);
            end
        end
    else
        if includeVcols
            fprintf(fid, 't,freq_hz,rocof_hz_s,v_pu_rms,dv_pu_dt_s\n');
            for k = 1:numel(tout)
                fprintf(fid, '%.12g,%.12g,%.12g,%.12g,%.12g\n', tout(k), freq(k), rocof(k), vpu(k), dvpu(k));
            end
        else
            fprintf(fid, 't,freq_hz,rocof_hz_s\n');
            for k = 1:numel(tout)
                fprintf(fid, '%.12g,%.12g,%.12g\n', tout(k), freq(k), rocof(k));
            end
        end
    end
    fclose(fid);
end

function tout = extract_tout(simOut)
    tout = [];
    if ~isa(simOut, 'Simulink.SimulationOutput')
        return;
    end
    try
        tout = simOut.tout;
    catch
        tout = [];
    end
end

function logs = extract_logsout(simOut)
    logs = [];
    if ~isa(simOut, 'Simulink.SimulationOutput')
        return;
    end
    try
        logs = simOut.logsout;
    catch
        logs = [];
    end
end

function idx = pick_log_element(logs, preferredSubstring)
    n = logs.numElements;
    idx = 1;
    if isempty(preferredSubstring)
        return;
    end
    ps = lower(char(preferredSubstring));
    for i = 1:n
        if ismethod(logs, 'getElement')
            nm = logs.getElement(i).Name;
        else
            nm = logs{i}.Name;
        end
        if ~isempty(strfind(lower(char(nm)), ps))
            idx = i;
            return;
        end
    end
end

function y_series = scalar_series_from_timeseries(ts)
    d = double(ts.Data);
    d = squeeze(d);
    if ndims(d) > 2
        d = reshape(d, size(d, 1), []);
    end
    if size(d, 2) > 1
        y_series = sqrt(mean(d .^ 2, 2));
    else
        y_series = d(:);
    end
end

function tf = is_likely_voltage_pu(y)
    y = abs(double(y(:)));
    if isempty(y)
        tf = false;
        return;
    end
    m = median(y);
    tf = m >= 0.5 && m <= 1.5 && max(y) < 2.5;
end

function hz = y_to_hz(y, f0_hz)
    y = double(y(:));
    if max(abs(y)) > 2 * f0_hz && max(abs(y)) > 100
        hz = y / (2 * pi);
    elseif max(abs(y)) < 5
        hz = f0_hz + y;
    else
        hz = y;
    end
end
