function [ TimeStr ] = disp_time( nowclock, verbose )
    if nargin<2
        verbose = 1;
    end
    if nargin<1
        nowclock = clock;
    end
    now_year = num2str(nowclock(1));
    if nowclock(2)<10
        now_month = ['0',num2str(nowclock(2))];
    else
        now_month = num2str(nowclock(2));
    end

    if nowclock(3)<10
        now_date = ['0',num2str(nowclock(3))];
    else
        now_date = num2str(nowclock(3));
    end

    if nowclock(4)<10
        now_hour = ['0',num2str(nowclock(4))];
    else
        now_hour = num2str(nowclock(4));
    end

    if nowclock(5)<10
        now_minute = ['0',num2str(nowclock(5))];
    else
        now_minute = num2str(nowclock(5));
    end

    if round(nowclock(6))<10
        now_second = ['0',num2str(round(nowclock(6)))];
    else
        now_second = num2str(round(nowclock(6)));
    end
    if verbose
        disp(['Time: ',now_year,'-',now_month,'-',now_date,...
            ' ',now_hour,':',now_minute,':',now_second]);
    end
    TimeStr=[now_year,'-',now_month,'-',now_date,' ',now_hour,':',now_minute,':',now_second];
end

