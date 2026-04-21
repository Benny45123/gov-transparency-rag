'use client';

import { motion } from 'framer-motion';

interface StatItem {
  label: string;
  value: string;
  icon?: string;
}

const defaultStats: StatItem[] = [
  { label: 'Docs', value: '2.1M+' },
  { label: 'Flights', value: '3,615' },
  { label: 'Emails', value: '1.78M' },
  { label: 'Persons', value: '1,508' },
  { label: 'Connections', value: '51K' },
];

export function StatsBar({ stats = defaultStats }: { stats?: StatItem[] }) {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.4, ease: 'easeOut' },
    },
  };

  return (
    <motion.div
      className="w-full flex flex-wrap gap-4 justify-center py-6"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {stats.map((stat) => (
        <motion.div
          key={stat.label}
          className="glass px-6 py-4 rounded-lg border border-amber-500/20 hover:border-amber-500/50 transition-colors"
          variants={itemVariants}
          whileHover={{ scale: 1.05 }}
        >
          <div className="text-amber-400 font-terminal font-bold text-lg">
            {stat.value}
          </div>
          <div className="text-gray-400 text-sm font-terminal uppercase tracking-wide">
            {stat.label}
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}
