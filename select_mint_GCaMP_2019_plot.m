% ANALYZE MEAN INTENSITY TRACES: RP/CB WINDOW SELECTION, THRESHOLD
% DEFINITION AND EVENT DETECTION

format compact
%% set parameters
clear;
fpath = 'C:\Users\ASUS\Documents\Hydra\Hydra_download_videos\\thermal\\thermaldatacsv'; % 数据文件夹路�?
fname = '30_CB_RP1_rawfloren'; % CSV 文件名（不带扩展名）
bkgname = ''; % '' if no bkg file
spath = fpath; % path to save result
figpath = fpath;
% 相机采集帧率（原始）
framerate_raw = 23.9; % 原始采集帧率（fps）
framerate = framerate_raw; % 实际有效帧率 = 23.9 fps

% 时间缩放倍数：将帧间秒数扩大 10 倍，对应真实物理时长
time_scale_factor = 10;
fprintf('Time scale factor: %d (multiply frame time by %d)\\n', time_scale_factor, time_scale_factor);

% 说明（中文注释）:
% 1. 该脚本用于分析 CB 和 RP1 两个通道的平均强度数据，计算 dF/F，并检测峰值。
% 2. 需要提供一个 CSV 文件，包含帧号、CB (peduncle)、RP1 (colume) 和刺激信息（可选）。
% 3. 脚本会自动识别列名，计算 dF/F  并使用 findpeaks 检测峰值位置，同时绘制两条曲线并标记峰值。
% 4. 最后会将分析结果保存为图像和 CSV 文件，CSV 文件包含时间、dF/F 值、刺激信息和峰值标记。

%% process data
% read data: �?CSV 读取多通道数据（帧号，CB，RP1，stimu）
csvfile = fullfile(fpath, [fname '.csv']);
if exist(csvfile, 'file')
    % 使用 readtable 读取 CSV，自动解析列名
    T = readtable(csvfile);
    colnames = T.Properties.VariableNames;
    
    % 查找列索引：frame, CB (peduncle), RP1 (colume), stim
    frame_idx = 1; % 第一列为帧号
    frames = table2array(T(:, frame_idx));
    
    cb_idx = []; rp1_idx = []; stim_idx = [];
    for i = 1:length(colnames)
        cname = colnames{i};
        if contains(cname, 'peduncle', 'IgnoreCase', true)
            cb_idx = i;
        elseif contains(cname, 'column', 'IgnoreCase', true)
            rp1_idx = i;
        elseif contains(cname, 'stimu', 'IgnoreCase', true)
            stim_idx = i;
        end
    end
    
    if isempty(cb_idx) || isempty(rp1_idx)
        error('Cannot find CB (peduncle) or RP1 (colume) columns in CSV.');
    end
    
    CB_raw = table2array(T(:, cb_idx))';
    RP1_raw = table2array(T(:, rp1_idx))';
    if ~isempty(stim_idx)
        stim = table2array(T(:, stim_idx))';
    else
        stim = ones(size(CB_raw)) * 30; % 默认无刺激
    end
    
    % 直接处理两个通道
    fprintf('Processing both CB and RP1 channels...\n');
else
    error('Cannot find %s', csvfile);
end
% 背景扣除（可选）
if ~isempty(bkgname)
    bkgcsv = fullfile(fpath, [bkgname '.csv']);
    if exist(bkgcsv, 'file')
        Tbkg = readtable(bkgcsv);
        bkg = table2array(Tbkg(:, 2))';
        mint = mint - bkg;
    end
end


% dF/F

window_seconds = 30; % 窗口长度（秒）
window_frames = max(1, round(window_seconds * framerate)); % 转为帧数，至少为 1

% 同时处理两个通道
fprintf('Computing dF/F for both channels with window: %d frames (%.1f s)\n', window_frames, window_seconds);
dff_CB = calc_dff_mw(CB_raw, window_frames);
dff_RP1 = calc_dff_mw(RP1_raw, window_frames);

% 计算一阶导数（用于检测峰值）
ddff_CB = gradient(dff_CB);
ddff_RP1 = gradient(dff_RP1);

numT = length(dff_CB);

% 生成时间向量
endtime_sec = (numT - 1) / framerate * time_scale_factor; % 秒（真实物理时间
t = linspace(0, endtime_sec, numT);

% 识别刺激区域：50 < stim < 100表示有刺激
stim_threshold_low = 50; % 根据实际数据调整刺激阈值
stim_threshold_high = 80; % 根据实际数据调整刺激阈值
stim_frames = find(stim > stim_threshold_low & stim < stim_threshold_high);

%% 使用 findpeaks 检测峰值并在同一图中绘制 CB 和 RP1 的 dF/F 曲线，标记峰值位置，并显示刺激窗
% 默认最小峰高可以根据需要调整，或者使用交互式输入来选择合适的阈值
minPeakHeight = 0; % 如果需要可调整或使用交互式输入
[pks_CB, locs_CB] = findpeaks(dff_CB, 'MinPeakHeight', minPeakHeight);
[pks_RP1, locs_RP1] = findpeaks(dff_RP1, 'MinPeakHeight', minPeakHeight);

% 统一 y 轴范围以便比较两条曲线
ysc_all = [min([dff_CB dff_RP1]) max([dff_CB dff_RP1])];
figure; set(gcf,'color','w','position',[255 336 1510 647])
hold on
% 绘制刺激窗并保存一个补丁句柄以供图例使用
hStim = []; % 初始值为 []，如果没有刺激则不显示
if ~isempty(stim_frames)
    stim_starts = stim_frames(find([1 diff(stim_frames)~=1]));
    stim_ends = stim_frames(find([diff(stim_frames)~=1 1]));
    for k = 1:length(stim_starts)
        stim_time_start = (stim_starts(k) - 1) / framerate * time_scale_factor;
        stim_time_end = (stim_ends(k) - 1) / framerate * time_scale_factor;
        hStim = patch([stim_time_start, stim_time_end, stim_time_end, stim_time_start], ...
            [ysc_all(1), ysc_all(1), ysc_all(2), ysc_all(2)], 'yellow', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end
% 绘制两条曲线并保存
hCB = plot(t, dff_CB, 'b', 'LineWidth', 1.5);
hRP1 = plot(t, dff_RP1, 'r', 'LineWidth', 1.5);
% 标记峰值并保存
hCBpk = plot(t(locs_CB), pks_CB, 'bo', 'MarkerFaceColor','b');
hRP1pk = plot(t(locs_RP1), pks_RP1, 'ro', 'MarkerFaceColor','r');

xlim([0 endtime_sec]); ylim(ysc_all);
xlabel('Time (seconds)'); ylabel('dF/F');
if isempty(hStim)
    legend([hCB,hRP1,hCBpk,hRP1pk],{'CB','RP1','CB peaks','RP1 peaks'}, 'Location','best');
else
    legend([hCB,hRP1,hCBpk,hRP1pk,hStim],{'CB','RP1','CB peaks','RP1 peaks','stim'}, 'Location','best');
end
title(sprintf('CB (blue) and RP1 (red) with peaks - Total duration: %.1f s', endtime_sec));
grid on;

% 保存合并
saveas(gcf, [figpath fname '_combined_analysis.fig']);
saveas(gcf, [figpath fname '_combined_analysis.png']);

%% 导出合并的数据为 CSV（含 stim 信息与峰值标记）
% 构建刺激标签18度或30度）
stim_label = repmat({''}, size(stim));
for i = 1:length(stim)
    if stim(i) <= 50
        stim_label{i} = '18';
    elseif stim(i) > 50
        stim_label{i} = '18';
    elseif ~isempty(stim)
        stim_label{i} = sprintf('%g', stim(i));
    end
end

% 生成峰值逻辑向量
peak_flag_CB = false(size(dff_CB));
peak_flag_CB(locs_CB) = true;
peak_flag_RP1 = false(size(dff_RP1));
peak_flag_RP1(locs_RP1) = true;

combined_table = table(t', dff_CB', dff_RP1', stim', stim_label', peak_flag_CB', peak_flag_RP1', ...
    'VariableNames', {'Time_seconds','dFF_CB','dFF_RP1','StimValue','StimLabel','CB_Peak','RP1_Peak'});
combined_csv = fullfile(spath, [fname '_analysis.csv']);
writetable(combined_table, combined_csv);
fprintf('Saved combined analysis to: %s\n', combined_csv);

% 简要输出刺激信息
if ~isempty(stim_frames)
    stim_starts_sec = (stim_starts - 1) / framerate * time_scale_factor;
    stim_ends_sec = (stim_ends - 1) / framerate * time_scale_factor;
    fprintf('Stim periods found: %d\n', length(stim_starts));
    for k = 1:length(stim_starts)
        fprintf('  Period %d: %.1f - %.1f s (duration: %.1f s)\n', k, stim_starts_sec(k), stim_ends_sec(k), stim_ends_sec(k) - stim_starts_sec(k));
    end
end

fprintf('\nAnalysis completed successfully!\n');
fprintf('Total duration: %.1f seconds (%.2f minutes)\n', endtime_sec, endtime_sec/60);

