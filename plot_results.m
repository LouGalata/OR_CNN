lear all
close all

baseline = load("results/baseline.mat").info;
baseline_v2 = load("results/baseline_v2.mat").info;

aug1 = load("results/augmentation=1.mat").info;
aug2 = load("results/augmentation=2.mat").info;
aug3 = load("results/augmentation=3.mat").info;
aug4 = load("results/augmentation=4.mat").info;
aug5 = load("results/augmentation=5.mat").info;

sam300 = load("results/samples=300-ef=13.3333.mat").info;
sam300_v2 = load("results/v2_samples=300-ef=13.3333.mat").info;

sam500 = load("results/samples=500-ef=8.mat").info;
sam1000 = load("results/samples=1000-ef=4.mat").info;
sam300_1 = load("results/samples=300-ef=1.mat").info;
sam500_1 = load("results/samples=500-ef=1.mat").info;
sam1000_1 = load("results/samples=1000-ef=1.mat").info;


bn_v1_300 = load("results/bv_v1_samples=300-ef=13.3333.mat").info;
bn_v1_1000 = load("results/bv_v1_samples=1000-ef=4.mat").info;
bn_v2_300 = load("results/bv_v1_samples=300-ef=13.3333.mat").info;

struct2csv(baseline, "results/baseline.csv")

x = 1:1:length(baseline.TrainingLoss);
x_300 = 1:1:length(sam300.TrainingLoss);
x_500 = 1:1:length(sam500.TrainingLoss);
x_1000 = 1:1:length(sam1000.TrainingLoss);

length(x)




% Print baseline and exp2 _________________________________________________

data = [baseline, sam300, sam500, sam1000];
colors = {[0 0 0], [81/255, 107/255, 127/255], [12/255, 132/255, 231/255], [48/255, 190/255, 163/255]};
labels = ["base_{train}", "base_{val}", "300_{train}", "300_{val}", "500_{train}",...
    "500_{val}", "1000_{train}", "1000_{val}"];

plot_results(data, colors, labels, "Exp2")

% Print baseline and exp2 less iteratos ___________________________________

data = [sam300, sam500, sam1000, sam300_1, sam500_1, sam1000_1];
colors = {[0 0 0], [81/255, 107/255, 127/255], [12/255, 132/255, 231/255], ...
    [48/255, 190/255, 163/255], [210/255, 98/255, 38/255 ], [116/255, 122/255, 255/255]};
labels = ["300_{train}", "300_{val}", "500_{train}",...
    "500_{val}", "1000_{train}", "1000_{val}", "300 EP=1_{train}", ...
    "300 EP=1_{val}", "500 EP=1_{train}",...
    "500 EP=1_{val}", "1000 EP=1_{train}", "1000 EP=1_{val}"];

plot_results(data, colors, labels, "Exp2_BN_v1_EP1")

% Print baseline and BN  _________________________________________________

data = [sam300, sam1000, bn_v1_300, bn_v1_1000];
colors = {[0 0 0], [81/255, 107/255, 127/255], [12/255, 132/255, 231/255], [116/255, 122/255, 255/255]};
labels = ["BN-300_{train}", "BN-300_{val}", "BN-1000_{train}", "BN-1000_{val}", ...
    "no-BN-300_{train}", "no-BN-300_{val}", "no-BN-1000_{train}", "no-BN-1000_{val}"];

plot_results(data, colors, labels, "Exp3_BN_v1")

% Print baseline and exp3 BN V2 ___________________________________________

data = [sam300_v2, bn_v2_300];
colors = {[0 0 0], [12/255, 132/255, 231/255]};
labels = ["BN-v2-300_{train}", "BN-v2-300 {val}", "no-BN-v2-300_{train}", "no-BN-v2-300 {val}"];

plot_results(data, colors, labels, "Exp3_BN_v2")

% Print baseline and exp4 Augmentation ____________________________________

data = [baseline, aug1, aug2, aug3, aug4, aug5];
colors = {[0 0 0], [12/255, 132/255, 231/255], [81/255, 107/255, 127/255], ...
    [210/255, 98/255, 38/255 ], [48/255, 190/255, 163/255], [116/255, 122/255, 255/255]};
labels = ["base_{train}", "base_{val}", "aug1_{val}", "aug1_{train}", ...
    "aug2_{val}", "aug2_{train}", "aug3_{val}", "aug3_{train}", ...
    "aug4_{val}", "aug4_{train}", "aug5_{val}", "aug5_{train}"];
    
plot_results(data, colors, labels, "Exp4_aug")




function plot_results(data, colors, labels, path)

    fig = figure('Renderer', 'painters', 'Position', [0 0 800 300]);
    t = tiledlayout(1,2,'TileSpacing','Compact','Padding','Compact');

    ax1 = nexttile;
    for i = 1:length(data)
        x = 1:1:length(data(i).TrainingLoss);
        train = data(i).TrainingLoss;
        val = data(i).ValidationLoss;
        idx = find(~isnan(data(i).ValidationLoss));
        colors(i)
        plot(x(idx), train(idx), 'Color', colors{i})
        hold(ax1, "on")
        plot(x(idx), val(idx), 'Color', colors{i}, 'Marker', '.', ...
            'LineStyle', '-', 'MarkerSize', 10)
    end
    grid on
    ax1.GridAlpha = 0.1;
    set(gca, 'YScale', 'log')
    title('Loss')
    
    ax2 = nexttile;
    for i = 1:length(data)
        x = 1:1:length(data(i).TrainingAccuracy);
        train = data(i).TrainingAccuracy;
        val = data(i).ValidationAccuracy;
        idx = find(~isnan(data(i).ValidationAccuracy));
        plot(x(idx), train(idx), 'Color', colors{i})
        hold(ax2, "on")
        plot(x(idx), val(idx), 'Color', colors{i}, 'Marker', '.', ...
            'LineStyle', '-', 'MarkerSize', 10)
    end
    grid on
    ax2.GridAlpha = 0.1;
    hold(ax2, "on")
    legend(labels, 'Location','southeast', 'FontSize', 8)

    title('Accuracy')
        
    saveas(gcf, "media/" + path + ".png")
end
